import fs from "node:fs/promises";
import path from "node:path";
import puppeteer from "../apps/web/node_modules/puppeteer/lib/esm/puppeteer/puppeteer.js";

const BASE_URL = process.env.LUCKYMODELS_BASE_URL || "http://127.0.0.1:38080";
const OUTPUT_DIR = process.env.LUCKYMODELS_SCREENSHOT_OUTPUT_DIR
  ? path.resolve(process.env.LUCKYMODELS_SCREENSHOT_OUTPUT_DIR)
  : path.resolve(
      "/home/liujingqian/LuckyModels/paper-v1.0/examples/hitbook/chinese/figures/luckymodels_screens",
    );

const ITEMS = [
  {
    path: "/benchmarks",
    file: "21_benchmarks_top.png",
    markerText: null,
    forbiddenTexts: ["Failed to fetch", "0 / 0 benchmark 记录", "暂无可对比的基准记录", "暂无论文表格数据"],
  },
  {
    path: "/benchmarks",
    file: "22_benchmarks_records.png",
    markerText: "基准记录",
    forbiddenTexts: ["Failed to fetch", "0 / 0 benchmark 记录"],
  },
  {
    path: "/benchmarks",
    file: "23_benchmarks_paper_view.png",
    markerText: "论文截图视图",
    forbiddenTexts: ["Failed to fetch", "暂无可对比的基准记录"],
  },
  {
    path: "/benchmarks",
    file: "24_benchmarks_paper_table.png",
    markerText: "论文表格视图",
    forbiddenTexts: ["Failed to fetch", "0 / 0 benchmark 记录", "暂无论文表格数据"],
  },
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loginAsAdmin(page) {
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
  await sleep(800);
  const hasLogin = await page.evaluate(() => document.body.innerText.includes("平台登录"));
  if (hasLogin) {
    await page.click('input[autocomplete="username"]', { clickCount: 3 });
    await page.keyboard.press("Backspace");
    await page.type('input[autocomplete="username"]', "admin", { delay: 40 });
    await page.click('input[autocomplete="current-password"]', { clickCount: 3 });
    await page.keyboard.press("Backspace");
    await page.type('input[autocomplete="current-password"]', "admin", { delay: 40 });
    await page.click('button[type="submit"]');
    await page.waitForFunction(() => !document.body.innerText.includes("平台登录"), { timeout: 15000 });
  }
  await page.waitForNetworkIdle({ idleTime: 1200, timeout: 30000 }).catch(() => {});
  await sleep(1000);
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
    target.scrollIntoView({ behavior: "instant", block: "start", inline: "nearest" });
    return true;
  }, markerText);

  if (!found) {
    await page.evaluate(() => window.scrollTo(0, 0));
  }
}

async function validatePageState(page, item) {
  const bodyText = await page.evaluate(() => document.body.innerText || "");
  const hit = (item.forbiddenTexts || []).find((text) => bodyText.includes(text));
  if (hit) {
    throw new Error(`found forbidden text: ${hit}`);
  }
}

async function captureItem(browser, item) {
  let lastError = null;
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    const page = await browser.newPage();
    try {
      await page.setExtraHTTPHeaders({
        "Accept-Language": "zh-CN,zh;q=0.9",
      });
      await loginAsAdmin(page);
      await page.goto(new URL(item.path, BASE_URL).toString(), { waitUntil: "domcontentloaded" });
      await sleep(2500);
      await page.waitForNetworkIdle({ idleTime: 1200, timeout: 30000 }).catch(() => {});
      await page.evaluate(async () => {
        if (document.fonts?.ready) {
          await document.fonts.ready;
        }
      });
      await sleep(800);
      await scrollToMarker(page, item.markerText);
      await sleep(1200);
      await validatePageState(page, item);
      await page.screenshot({
        path: path.join(OUTPUT_DIR, item.file),
        fullPage: false,
      });
      console.log(`captured: ${item.file}`);
      return;
    } catch (error) {
      lastError = error;
      console.error(`retry ${attempt}/4: ${item.file} -> ${error.message}`);
    } finally {
      await page.close();
    }
  }
  throw lastError;
}

async function main() {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
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
    for (const item of ITEMS) {
      await captureItem(browser, item);
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
