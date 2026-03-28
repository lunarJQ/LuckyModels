export async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

export async function loadDashboardData() {
  const [summary, models, deployments, benchmarks, report] = await Promise.all([
    fetchJSON("/api/v1/system/summary"),
    fetchJSON("/api/v1/models"),
    fetchJSON("/api/v1/deployments"),
    fetchJSON("/api/v1/benchmarks"),
    fetchJSON("/api/v1/demo/casrel/report"),
  ]);

  const modelDetailsEntries = await Promise.all(
    models.map(async (model) => {
      const detail = await fetchJSON(`/api/v1/models/${encodeURIComponent(model.model_id)}`);
      return [model.model_id, detail];
    }),
  );

  return {
    summary,
    models,
    deployments,
    benchmarks,
    report,
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

export async function getDeploymentHealth(deploymentId) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}/health`);
}

export async function getDeploymentLogs(deploymentId) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}/logs`);
}

export async function predictWithDeployment(deploymentId, payload) {
  return fetchJSON(`/api/v1/deployments/${encodeURIComponent(deploymentId)}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
