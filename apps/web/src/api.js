export async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

export async function fetchBlob(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return {
    blob: await response.blob(),
    filename: parseFilenameFromDisposition(response.headers.get("content-disposition")),
  };
}

export function xhrJSON(url, options = {}) {
  return xhrRequest(url, { ...options, responseType: "json" });
}

export function xhrBlob(url, options = {}) {
  return xhrRequest(url, { ...options, responseType: "blob" });
}

export async function fetchAuthSession() {
  return fetchJSON("/api/v1/auth/session");
}

export async function loginWithPassword(payload) {
  return fetchJSON("/api/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function logoutSession() {
  return fetchJSON("/api/v1/auth/logout", {
    method: "POST",
  });
}

export async function loadDashboardData() {
  const [summary, models, deployments, benchmarks, predictionArchives, report, capabilityMatrix, gpuStatus, accessOverview, importOptions] = await Promise.all([
    fetchJSON("/api/v1/system/summary"),
    fetchJSON("/api/v1/models"),
    fetchJSON("/api/v1/deployments"),
    fetchJSON("/api/v1/benchmarks"),
    fetchJSON("/api/v1/prediction-archives"),
    fetchJSON("/api/v1/demo/casrel/report"),
    fetchJSON("/api/v1/models/capability-matrix"),
    fetchJSON("/api/v1/system/gpus"),
    fetchJSON("/api/v1/access-control/overview"),
    fetchJSON("/api/v1/models/import-options"),
  ]);

  const deploymentEntries = await Promise.all(
    deployments.map(async (deployment) => {
      if (!deployment.service_url || !["running", "starting"].includes(deployment.status)) {
        return deployment;
      }
      try {
        const health = await getDeploymentHealth(deployment.deployment_id);
        return { ...deployment, health };
      } catch (error) {
        return {
          ...deployment,
          health: {
            deployment_id: deployment.deployment_id,
            reachable: false,
            pid_alive: false,
            detail: error instanceof Error ? error.message : String(error),
          },
        };
      }
    }),
  );

  const modelDetailsEntries = await Promise.all(
    models.map(async (model) => {
      const detail = await fetchJSON(`/api/v1/models/${encodeURIComponent(model.model_id)}`);
      return [model.model_id, detail];
    }),
  );

  return {
    summary,
    models,
    deployments: deploymentEntries,
    benchmarks,
    predictArchives: predictionArchives,
    report,
    capabilityMatrix,
    gpuStatus,
    accessOverview,
    importOptions,
    modelDetails: Object.fromEntries(modelDetailsEntries),
  };
}

export async function runCasrelDemo() {
  return fetchJSON("/api/v1/demo/casrel/run", { method: "POST" });
}

export async function createDeployment(payload) {
  return fetchJSON("/api/v1/deployments", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function stopDeployment(deploymentId) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}/stop`, {
    method: "POST",
  });
}

export async function restartDeployment(deploymentId) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}/restart`, {
    method: "POST",
  });
}

export async function deleteDeployment(deploymentId) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}`, {
    method: "DELETE",
  });
}

export async function createBenchmark(payload) {
  return fetchJSON("/api/v1/benchmarks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteBenchmark(benchmarkId) {
  return fetchJSON(`/api/v1/benchmarks/${encodeURIComponent(benchmarkId)}`, {
    method: "DELETE",
  });
}

export async function deletePredictionArchive(archiveId) {
  return fetchJSON(`/api/v1/prediction-archives/${encodeURIComponent(archiveId)}`, {
    method: "DELETE",
  });
}

export async function getDeploymentHealth(deploymentId) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}/health`);
}

export async function getDeploymentLogs(deploymentId) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}/logs`);
}

export async function getModelProfiles(modelId) {
  return fetchJSON(`/api/v1/models/${encodeURIComponent(modelId)}/profiles`);
}

export async function importModelArchive(payload) {
  const formData = new FormData();
  formData.append("archive", payload.archive);
  formData.append("import_mode", payload.importMode);
  formData.append("model_family", payload.modelFamily);
  if (payload.language) formData.append("language", payload.language);
  if (payload.datasetName) formData.append("dataset_name", payload.datasetName);
  if (payload.version) formData.append("version", payload.version);
  if (payload.modelName) formData.append("model_name", payload.modelName);
  if (payload.backbone) formData.append("backbone", payload.backbone);
  if (payload.alias) formData.append("alias", payload.alias);
  return xhrJSON("/api/v1/models/import", {
    method: "POST",
    body: formData,
    onUploadProgress: payload.onUploadProgress,
  });
}

export async function exportModelPackage(modelId, onDownloadProgress) {
  return xhrBlob(`/api/v1/models/${encodeURIComponent(modelId)}/export`, {
    method: "GET",
    onDownloadProgress,
  });
}

export async function predictWithDeployment(deploymentId, payload) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function runTaskRoutingStrategy(payload) {
  return fetchJSON("/api/v1/strategies/task-routing/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getDeploymentPolicy() {
  return fetchJSON("/api/v1/strategies/deployment-policy");
}

export async function getDeploymentPolicyAutomation() {
  return fetchJSON("/api/v1/strategies/deployment-policy/automation");
}

export async function updateDeploymentPolicyAutomation(payload) {
  return fetchJSON("/api/v1/strategies/deployment-policy/automation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function runDeploymentPolicyAutomationNow() {
  return fetchJSON("/api/v1/strategies/deployment-policy/automation/run", {
    method: "POST",
  });
}

export async function createAccessUser(payload) {
  return fetchJSON("/api/v1/access-control/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function updateAccessUser(userId, payload) {
  return fetchJSON(`/api/v1/access-control/users/${encodeURIComponent(userId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function toggleAccessUser(userId, payload) {
  return fetchJSON(`/api/v1/access-control/users/${encodeURIComponent(userId)}/toggle`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function switchCurrentAccessUser(payload) {
  return fetchJSON("/api/v1/access-control/current-user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function resetAccessUserPassword(userId, payload) {
  return fetchJSON(`/api/v1/access-control/users/${encodeURIComponent(userId)}/reset-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function parseFilenameFromDisposition(value) {
  const text = String(value || "");
  const utf8Match = text.match(/filename\*\s*=\s*UTF-8''([^;]+)/i);
  if (utf8Match) {
    return decodeURIComponent(utf8Match[1]);
  }
  const plainMatch = text.match(/filename\s*=\s*"([^"]+)"/i) || text.match(/filename\s*=\s*([^;]+)/i);
  if (!plainMatch) {
    return "";
  }
  return plainMatch[1].trim();
}

function xhrRequest(url, options = {}) {
  const {
    method = "GET",
    body,
    headers = {},
    responseType = "json",
    onUploadProgress,
    onDownloadProgress,
  } = options;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url, true);
    xhr.withCredentials = true;
    if (responseType === "blob") {
      xhr.responseType = "blob";
    }

    Object.entries(headers).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        xhr.setRequestHeader(key, value);
      }
    });

    xhr.upload.onprogress = (event) => {
      if (typeof onUploadProgress === "function") {
        onUploadProgress(toProgressPayload(event));
      }
    };

    xhr.onprogress = (event) => {
      if (typeof onDownloadProgress === "function") {
        onDownloadProgress(toProgressPayload(event));
      }
    };

    xhr.onerror = () => {
      reject(new Error("网络请求失败"));
    };

    xhr.onabort = () => {
      reject(new Error("请求已取消"));
    };

    xhr.onload = () => {
      const ok = xhr.status >= 200 && xhr.status < 300;
      if (!ok) {
        const text = xhr.responseType === "blob" ? "" : xhr.responseText;
        reject(new Error(`${xhr.status} ${xhr.statusText}: ${text}`));
        return;
      }

      try {
        if (responseType === "blob") {
          resolve({
            blob: xhr.response,
            filename: parseFilenameFromDisposition(xhr.getResponseHeader("content-disposition")),
          });
          return;
        }
        const text = xhr.responseText || "";
        resolve(text ? JSON.parse(text) : null);
      } catch (error) {
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    };

    xhr.send(body);
  });
}

function toProgressPayload(event) {
  const total = event.lengthComputable ? event.total : null;
  const loaded = event.loaded ?? 0;
  return {
    loaded,
    total,
    lengthComputable: Boolean(event.lengthComputable),
    percent: total && total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : null,
  };
}
