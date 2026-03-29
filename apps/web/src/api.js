export async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
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
  const [summary, models, deployments, benchmarks, predictionArchives, report, capabilityMatrix, gpuStatus, accessOverview] = await Promise.all([
    fetchJSON("/api/v1/system/summary"),
    fetchJSON("/api/v1/models"),
    fetchJSON("/api/v1/deployments"),
    fetchJSON("/api/v1/benchmarks"),
    fetchJSON("/api/v1/prediction-archives"),
    fetchJSON("/api/v1/demo/casrel/report"),
    fetchJSON("/api/v1/models/capability-matrix"),
    fetchJSON("/api/v1/system/gpus"),
    fetchJSON("/api/v1/access-control/overview"),
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
