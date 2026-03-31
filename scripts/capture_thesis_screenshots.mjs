import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = process.env.LUCKYMODELS_BASE_URL || "http://127.0.0.1:38080";
const OUTPUT_DIR = process.env.LUCKYMODELS_SCREENSHOT_OUTPUT_DIR
  ? path.resolve(process.env.LUCKYMODELS_SCREENSHOT_OUTPUT_DIR)
  : path.resolve(
      __dirname,
      "../paper-v1.0/examples/hitbook/chinese/figures/luckymodels_screens",
    );
const SCREENSHOT_FILTER = (process.env.LUCKYMODELS_SCREENSHOT_FILTER || "").trim();
const PUPPETEER_ENTRY = path.resolve(
  __dirname,
  "../apps/web/node_modules/puppeteer/lib/esm/puppeteer/puppeteer.js",
);
const FORCE_RECAPTURE = process.env.LUCKYMODELS_SCREENSHOT_FORCE === "1";

const SCREENSHOTS = [
  {
    path: "/",
    file: "01_overview.png",
    title: "资产概览",
    forbiddenTexts: ["Failed to fetch", "加载失败"],
  },
  { path: "/models", file: "02_models_top.png", title: "模型库总览" },
  { path: "/models", file: "03_models_import.png", title: "模型导入区", markerText: "上传模型制品" },
  { path: "/models", file: "04_models_assets.png", title: "模型资产区", markerText: "模型资产" },
  { path: "/capability-matrix", file: "05_capability_matrix_top.png", title: "能力矩阵总览" },
  { path: "/capability-matrix", file: "06_capability_matrix_table.png", title: "能力矩阵总表", markerText: "固定能力总表" },
  { path: "/artifacts", file: "07_artifacts_list.png", title: "制品包清单" },
  { path: "/deployments", file: "08_deployments_top.png", title: "部署记录总览" },
  { path: "/deployments", file: "09_deployments_control.png", title: "部署控制区", markerText: "部署控制" },
  { path: "/deployments", file: "10_deployments_health.png", title: "部署健康与日志", markerText: "健康状态" },
  { path: "/experiments", file: "11_experiments_top.png", title: "在线推理总览" },
  { path: "/experiments", file: "12_experiments_result.png", title: "在线推理结果", markerText: "预测结果" },
  { path: "/experiments", file: "13_experiments_archive.png", title: "在线推理归档", markerText: "单次预测归档" },
  { path: "/task-routing", file: "14_task_routing_top.png", title: "任务调度总览" },
  { path: "/task-routing", file: "15_task_routing_result.png", title: "任务调度结果", markerText: "调度结果" },
  { path: "/task-routing", file: "16_task_routing_candidates.png", title: "候选部署与执行明细", markerText: "候选部署与执行明细" },
  { path: "/task-routing", file: "17_task_routing_history.png", title: "最近决策记录", markerText: "最近决策记录" },
  { path: "/deployment-policy", file: "18_deployment_policy_top.png", title: "部署策略总览" },
  { path: "/deployment-policy", file: "19_deployment_policy_gpu.png", title: "GPU 资源视图", markerText: "当前资源视图" },
  { path: "/deployment-policy", file: "20_deployment_policy_table.png", title: "Deployment 优化方案表", markerText: "现有 Deployment 优化方案表" },
  {
    path: "/benchmarks",
    file: "21_benchmarks_top.png",
    title: "基准剖析总览",
    forbiddenTexts: ["Failed to fetch", "0 / 0 benchmark 记录", "暂无可对比的基准记录", "暂无论文表格数据"],
  },
  {
    path: "/benchmarks",
    file: "22_benchmarks_records.png",
    title: "基准记录",
    markerText: "基准记录",
    forbiddenTexts: ["Failed to fetch", "0 / 0 benchmark 记录"],
  },
  {
    path: "/benchmarks",
    file: "23_benchmarks_paper_view.png",
    title: "论文截图视图",
    markerText: "论文截图视图",
    forbiddenTexts: ["Failed to fetch", "暂无可对比的基准记录"],
  },
  {
    path: "/benchmarks",
    file: "24_benchmarks_paper_table.png",
    title: "论文表格视图",
    markerText: "论文表格视图",
    forbiddenTexts: ["Failed to fetch", "0 / 0 benchmark 记录", "暂无论文表格数据"],
  },
  { path: "/resources", file: "25_resources_gpu.png", title: "运行资源与 GPU 状态" },
  { path: "/resources", file: "26_resources_runtime.png", title: "运行实例快照", markerText: "运行实例快照" },
  { path: "/logs", file: "27_logs.png", title: "日志中心" },
  { path: "/archive", file: "28_archive.png", title: "结果归档" },
  { path: "/access-control", file: "29_access_control_top.png", title: "用户权限总览" },
  { path: "/access-control", file: "30_access_control_matrix.png", title: "用户与权限矩阵", markerText: "用户与权限矩阵" },
  { path: "/admin-console", file: "31_admin_console_top.png", title: "管理员控制台" },
  { path: "/admin-console", file: "32_admin_console_governance.png", title: "账号治理", markerText: "角色调整、账号启停与密码重置" },
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadPuppeteer() {
  const moduleUrl = pathToFileURL(PUPPETEER_ENTRY).href;
  const imported = await import(moduleUrl);
  return imported.default || imported;
}

async function loginAsAdmin(page) {
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
  await sleep(800);
  const hasLogin = await page.evaluate(() => document.body.innerText.includes("平台登录"));
  if (!hasLogin) {
    return;
  }

  await page.click('input[autocomplete="username"]', { clickCount: 3 });
  await page.keyboard.press("Backspace");
  await page.type('input[autocomplete="username"]', "admin", { delay: 40 });

  await page.click('input[autocomplete="current-password"]', { clickCount: 3 });
  await page.keyboard.press("Backspace");
  await page.type('input[autocomplete="current-password"]', "admin", { delay: 40 });

  await page.click('button[type="submit"]');
  await page.waitForFunction(
    () => !document.body.innerText.includes("平台登录"),
    { timeout: 15000 },
  );
  await sleep(1200);
}

async function ensureLoggedIn(page) {
  await loginAsAdmin(page);
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
  await sleep(1500);
  await page.waitForNetworkIdle({ idleTime: 1200, timeout: 30000 }).catch(() => {});
  await sleep(800);
}

async function scrollToMarker(page, markerText) {
  if (!markerText) {
    await page.evaluate(() => window.scrollTo(0, 0));
    return;
  }

  const found = await page.evaluate((text) => {
    const nodes = Array.from(document.querySelectorAll("h1, h2, h3, h4, .section-header, .card, .page-shell, strong, div, span"));
    const target = nodes
      .filter((node) => {
        const value = (node.textContent || "").replace(/\s+/g, " ").trim();
        return value.includes(text);
      })
      .sort((left, right) => {
        const leftLength = (left.textContent || "").trim().length;
        const rightLength = (right.textContent || "").trim().length;
        return leftLength - rightLength;
      })[0];
    if (!target) {
      return false;
    }
    const value = (target.textContent || "").replace(/\s+/g, " ").trim();
    if (!value.includes(text)) {
      return false;
    }
    target.scrollIntoView({ behavior: "instant", block: "start", inline: "nearest" });
    return true;
  }, markerText);

  if (!found) {
    await page.evaluate(() => window.scrollTo(0, 0));
  }
}

async function captureRoute(page, item) {
  const targetUrl = new URL(item.path, BASE_URL).toString();
  await page.goto(targetUrl, { waitUntil: "domcontentloaded" });
  await sleep(2500);
  await page.waitForNetworkIdle({ idleTime: 1200, timeout: 30000 }).catch(() => {});
  await page.evaluate(async () => {
    if (document.fonts?.ready) {
      await document.fonts.ready;
    }
  });
  await sleep(500);

  const hasLogin = await page.evaluate(() => document.body.innerText.includes("平台登录"));
  if (hasLogin) {
    await ensureLoggedIn(page);
    await page.goto(targetUrl, { waitUntil: "domcontentloaded" });
    await sleep(2500);
    await page.waitForNetworkIdle({ idleTime: 1200, timeout: 30000 }).catch(() => {});
    await page.evaluate(async () => {
      if (document.fonts?.ready) {
        await document.fonts.ready;
      }
    });
    await sleep(500);
  }

  await scrollToMarker(page, item.markerText);
  await sleep(1200);
  await validatePageState(page, item);
  await page.screenshot({
    path: path.join(OUTPUT_DIR, item.file),
    fullPage: false,
  });
}

async function validatePageState(page, item) {
  const bodyText = await page.evaluate(() => document.body.innerText || "");
  const forbiddenTexts = item.forbiddenTexts || [];
  const hit = forbiddenTexts.find((text) => bodyText.includes(text));
  if (hit) {
    throw new Error(`page validation failed for ${item.file}: found forbidden text "${hit}"`);
  }
}

async function captureWithRetry(page, item, retries = 3) {
  const outputPath = path.join(OUTPUT_DIR, item.file);
  if (!FORCE_RECAPTURE) {
    try {
      await fs.access(outputPath);
      console.log(`skip: ${item.file}`);
      return;
    } catch {
      // continue
    }
  }

  let lastError;
  for (let attempt = 1; attempt <= retries; attempt += 1) {
    try {
      await captureRoute(page, item);
      console.log(`captured: ${item.file} (${item.title})`);
      return;
    } catch (error) {
      lastError = error;
      console.error(`retry ${attempt}/${retries}: ${item.file} -> ${error.message}`);
      await sleep(1500 * attempt);
      await ensureLoggedIn(page);
    }
  }
  throw lastError;
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  const puppeteer = await loadPuppeteer();
  const screenshotItems = SCREENSHOT_FILTER
    ? SCREENSHOTS.filter((item) => item.file.includes(SCREENSHOT_FILTER) || item.path.includes(SCREENSHOT_FILTER))
    : SCREENSHOTS;

  if (SCREENSHOT_FILTER && screenshotItems.length === 0) {
    throw new Error(`no screenshots matched filter: ${SCREENSHOT_FILTER}`);
  }

  const browser = await puppeteer.launch({
    headless: true,
    defaultViewport: {
      width: 1600,
      height: 980,
      deviceScaleFactor: 1.25,
    },
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  try {
    for (const item of screenshotItems) {
      const page = await browser.newPage();
      try {
        await page.setExtraHTTPHeaders({
          "Accept-Language": "zh-CN,zh;q=0.9",
        });
        await ensureLoggedIn(page);
        await captureWithRetry(page, item);
      } finally {
        await page.close();
      }
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
