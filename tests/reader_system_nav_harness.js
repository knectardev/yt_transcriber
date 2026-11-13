/**
 * Node harness for reader system-nav and reading-position helpers.
 * Run: node tests/reader_system_nav_harness.js
 */

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function runExported(relPath, names, sandboxExtra = {}) {
  const file = path.join(__dirname, "..", "reader", "js", relPath);
  const source = fs.readFileSync(file, "utf8").replace(/^export /gm, "");
  const bag = {};
  const assign = names.map((name) => `bag.${name} = ${name};`).join("\n");
  vm.runInNewContext(`${source}\n${assign}`, { bag, console, ...sandboxExtra });
  return bag;
}

function testSystemNav() {
  const bag = runExported("system-nav.js", [
    "systemsFromBoxes",
    "systemIndexForMeasure",
    "scrollTopForSystem",
  ]);
  const boxes = [
    { measure: 1, systemKey: "1:0", minY: 0, maxY: 10, scale: 10 },
    { measure: 2, systemKey: "1:0", minY: 1, maxY: 11, scale: 10 },
    { measure: 3, systemKey: "1:1", minY: 20, maxY: 30, scale: 10 },
    { measure: 4, systemKey: "1:1", minY: 21, maxY: 31, scale: 10 },
  ];
  const systems = bag.systemsFromBoxes(boxes);
  assert.strictEqual(systems.length, 2);
  assert.strictEqual(systems[0].anchorMeasure, 1);
  assert.strictEqual(systems[1].anchorMeasure, 3);
  assert.strictEqual(bag.systemIndexForMeasure(systems, 4), 1);
  assert.strictEqual(bag.systemIndexForMeasure(systems, 1), 0);
  const top = bag.scrollTopForSystem(systems, 1, 0.5, 0);
  assert.strictEqual(top, 145);
  console.log("system-nav: ok");
}

function testPosition() {
  const store = new Map();
  const window = {
    localStorage: {
      getItem: (key) => (store.has(key) ? store.get(key) : null),
      setItem: (key, value) => store.set(key, String(value)),
      removeItem: (key) => store.delete(key),
    },
  };
  const bag = runExported(
    "position.js",
    ["loadReadingPosition", "saveReadingPosition", "clearReadingPosition", "POSITION_KEY"],
    { window }
  );
  bag.saveReadingPosition("catalog:bwv-0772", 12);
  const loaded = bag.loadReadingPosition();
  assert.strictEqual(loaded.pieceId, "catalog:bwv-0772");
  assert.strictEqual(loaded.anchorMeasure, 12);
  bag.clearReadingPosition();
  assert.strictEqual(bag.loadReadingPosition(), null);
  console.log("position: ok");
}

testSystemNav();
testPosition();
console.log("reader harness passed");
