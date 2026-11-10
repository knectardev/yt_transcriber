// Runs the script of a generated playback page outside a browser.
//
//   node playback_page_harness.js <page.html> <scenario.js> [query string]
//
// The page's own inline script is executed unchanged in a fresh context with
// a small stand-in for the document, a player that records what it is asked
// to do, and a clock the scenario controls. The scenario is JavaScript that
// runs in the same context, so it can call the page's functions and read its
// state. Its last expression is printed as JSON.
//
// The scenario gets one object, H:
//   H.player            the stand-in player: time, state (1 playing, 2 paused), seeks[], cued
//   H.el(id)            the stand-in element with that id
//   H.click(id)         fire that element's click listeners
//   H.change(id, on)    set a checkbox and fire its change listeners
//   H.draw()            give the page a drawn score and build its measure hits
//   H.clickMeasure(n)   click the hit of measure n
//   H.selected()        measure numbers whose hit carries the "selected" class
//   H.active()          measure number of the highlighted hit, or null
//   H.at(seconds)       put the player at a video time, let a second pass, and run one tick
//   H.posts             bodies the page sent with fetch(..., {method: "POST"})

const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const scenario = fs.readFileSync(process.argv[3], "utf8");
const search = process.argv[4] || "";

const inline = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((match) => match[1]);
const source = inline.find((text) => text.includes("const config ="));
if (!source) {
  console.error("The page has no inline script with a config.");
  process.exit(2);
}

// Attributes the markup starts with, so a checkbox that is ticked in the
// page is ticked here, and a link that is hidden in the page is hidden here.
function startsWith(attribute) {
  const found = new Set();
  const pattern = new RegExp('<[a-z0-9]+\\b[^>]*\\bid="([^"]+)"[^>]*\\b' + attribute + "\\b[^>]*>", "gi");
  for (const match of html.matchAll(pattern)) found.add(match[1]);
  return found;
}
const checkedAtStart = startsWith("checked");
const hiddenAtStart = startsWith("hidden");

const created = [];
const byId = {};

function element(id) {
  const classes = new Set();
  const listeners = {};
  const node = {
    id: id || "",
    className: "",
    hidden: id ? hiddenAtStart.has(id) : false,
    checked: id ? checkedAtStart.has(id) : false,
    textContent: "",
    href: "",
    title: "",
    dataset: {},
    style: {},
    attributes: {},
    removed: false,
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, on) => {
        const want = on === undefined ? !classes.has(name) : !!on;
        if (want) classes.add(name);
        else classes.delete(name);
        return want;
      },
    },
    addEventListener: (type, handler) => {
      (listeners[type] = listeners[type] || []).push(handler);
    },
    fire: (type) => {
      (listeners[type] || []).forEach((handler) => handler({ currentTarget: node, target: node, stopPropagation() {} }));
    },
    setAttribute: (name, value) => {
      node.attributes[name] = String(value);
    },
    getAttribute: (name) => node.attributes[name],
    appendChild: (child) => child,
    insertBefore: (child) => child,
    remove: () => {
      node.removed = true;
    },
    querySelector: () => null,
    getBoundingClientRect: () => ({ top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 }),
  };
  return node;
}

function el(id) {
  if (!byId[id]) byId[id] = element(id);
  return byId[id];
}

function hits() {
  return created.filter((node) => node.className === "measure-hit" && !node.removed);
}

const clock = { now: 0 };
const posts = [];
const player = {
  time: 0,
  state: 2,
  rate: 1,
  length: 3600,
  seeks: [],
  cued: null,
  getCurrentTime: () => player.time,
  getPlayerState: () => player.state,
  getPlaybackRate: () => player.rate,
  getDuration: () => player.length,
  seekTo: (seconds) => {
    player.seeks.push(seconds);
    player.time = seconds;
  },
  playVideo: () => {
    player.state = 1;
  },
  pauseVideo: () => {
    player.state = 2;
  },
  cueVideoById: (request) => {
    player.cued = request;
  },
};

const context = {
  console,
  URLSearchParams,
  document: {
    body: element("body"),
    getElementById: el,
    createElement: () => {
      const node = element("");
      created.push(node);
      return node;
    },
    createTextNode: (text) => ({ textContent: text }),
    querySelectorAll: (selector) => (selector === ".measure-hit" ? hits() : []),
    querySelector: (selector) =>
      selector === ".measure-hit.active" ? hits().find((node) => node.classList.contains("active")) || null : null,
  },
  location: { search: search },
  localStorage: { getItem: () => null, setItem: () => {} },
  performance: { now: () => clock.now },
  requestAnimationFrame: (callback) => callback(),
  setInterval: () => 0,
  fetch: (address, options) => {
    if (options && options.method === "POST") posts.push({ address: address, body: JSON.parse(options.body) });
    // The page's opening request for the stored offset gets no answer here.
    return Promise.reject(new Error("no server"));
  },
  scrollY: 0,
  scrollTo: () => {},
};
context.window = context;
vm.createContext(context);
vm.runInContext(source, context, { filename: "page-script.js" });

// One box per anchored measure, laid out left to right, as the drawing library reports them.
function drawnScore() {
  const measures = vm.runInContext("anchors.map(function (anchor) { return anchor.measure; })", context);
  return {
    Zoom: 1,
    GraphicSheet: {
      MusicPages: [],
      MeasureList: measures.map((number, index) => [
        {
          PositionAndShape: {
            AbsolutePosition: { x: index * 20, y: 0 },
            BorderLeft: 0,
            BorderTop: 0,
            Size: { width: 20, height: 10 },
          },
          parentSourceMeasure: { MeasureNumber: number },
          ParentStaffLine: null,
        },
      ]),
    },
  };
}

context.H = {
  player: player,
  posts: posts,
  clock: clock,
  el: el,
  click: (id) => el(id).fire("click"),
  change: (id, on) => {
    el(id).checked = !!on;
    el(id).fire("change");
  },
  draw: () => {
    context.__score = drawnScore();
    vm.runInContext("osmd = __score; buildMeasureHits();", context);
  },
  clickMeasure: (number) => {
    const hit = hits().find((node) => Number(node.dataset.measure) === number);
    if (!hit) throw new Error("No hit for measure " + number);
    hit.fire("click");
  },
  selected: () =>
    hits()
      .filter((node) => node.classList.contains("selected"))
      .map((node) => Number(node.dataset.measure)),
  active: () => {
    const hit = hits().find((node) => node.classList.contains("active"));
    return hit ? Number(hit.dataset.measure) : null;
  },
  at: (seconds) => {
    player.time = seconds;
    clock.now += 1000;
    vm.runInContext("syncFromPlayer();", context);
  },
};
vm.runInContext("player = H.player;", context);

const result = vm.runInContext(scenario, context, { filename: "scenario.js" });
process.stdout.write(JSON.stringify(result === undefined ? null : result));
