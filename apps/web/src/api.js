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
