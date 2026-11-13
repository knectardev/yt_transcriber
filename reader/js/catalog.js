/** Load the published catalog index, full library metadata, and score packages. */

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

export async function loadCatalogLibrary(baseUrl = ".") {
  const response = await fetch(join(baseUrl, "catalog/library.json"), { cache: "no-cache" });
  if (!response.ok) {
    return { schemaVersion: 1, entries: [], updatedAt: null };
  }
  const data = await response.json();
  return {
    schemaVersion: data.schemaVersion || 1,
    updatedAt: data.updatedAt || null,
    entries: Array.isArray(data.entries) ? data.entries : [],
  };
}

/** Merge full Bach metadata with published package readiness. */
export function mergeCatalogViews(library, published) {
  const publishedById = new Map();
  for (const row of published.entries || []) {
    publishedById.set(row.entryId, row);
  }
  const libraryEntries = library.entries || [];
  if (!libraryEntries.length) {
    return (published.entries || []).map((row) => ({
      ...row,
      published: true,
      status: "published",
    }));
  }
  return libraryEntries.map((item) => {
    const pub = publishedById.get(item.entryId);
    if (!pub) {
      return {
        ...item,
        published: false,
        status: "library",
        sections: [],
      };
    }
    return {
      ...item,
      ...pub,
      published: true,
      status: "published",
      sections: Array.isArray(pub.sections) ? pub.sections : [],
    };
  });
}

export async function loadPackage(entryOrSection, baseUrl = ".") {
  const path =
    entryOrSection.packagePath ||
    (entryOrSection.movementId
      ? `catalog/packages/${entryOrSection.entryId}__${entryOrSection.movementId}.json`
      : `catalog/packages/${entryOrSection.entryId}.json`);
  const response = await fetch(join(baseUrl, path), { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Could not load package for ${entryOrSection.entryId || entryOrSection.id}.`);
  }
  return response.json();
}

function join(base, path) {
  const root = String(base || ".").replace(/\/+$/, "");
  const rel = String(path || "").replace(/^\/+/, "");
  return `${root}/${rel}`;
}
