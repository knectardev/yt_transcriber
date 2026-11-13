/** Load the published catalog index and score packages. */

export async function loadCatalogIndex(baseUrl = ".") {
  const response = await fetch(join(baseUrl, "catalog/index.json"), { cache: "no-cache" });
  if (!response.ok) {
    throw new Error("Could not load published catalog index.");
  }
  const data = await response.json();
  const entries = Array.isArray(data.entries) ? data.entries : [];
  return {
    schemaVersion: data.schemaVersion || 1,
    updatedAt: data.updatedAt || null,
    entries,
  };
}

export async function loadPackage(entry, baseUrl = ".") {
  const path = entry.packagePath || `catalog/packages/${entry.entryId}.json`;
  const response = await fetch(join(baseUrl, path), { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Could not load package for ${entry.entryId || entry.id}.`);
  }
  return response.json();
}

function join(base, path) {
  const root = String(base || ".").replace(/\/+$/, "");
  const rel = String(path || "").replace(/^\/+/, "");
  return `${root}/${rel}`;
}
