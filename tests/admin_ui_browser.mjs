// Browser regressions for the embedded UI. Uses an isolated in-memory API, never production.
// Run: NODE_PATH=<installed Playwright node_modules> node tests/admin_ui_browser.mjs
// Preview fixtures: node tests/admin_ui_browser.mjs --serve (127.0.0.1:9127).
import assert from "node:assert/strict";
import { readFileSync, mkdirSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(path.join(root, "src/admin.py"), "utf8")
  .split('_ADMIN_HTML = r"""')[1]
  .split('"""')[0];
const now = Math.floor(Date.now() / 1000);
const summary = {
  requests: 3,
  ok: 2,
  errors: 1,
  input_tokens: 1200,
  output_tokens: 300,
  cached_read_tokens: 500,
  cost_usd: 0.032,
  known_cost_requests: 2,
  unknown_cost_requests: 1,
  usage_reported_requests: 3,
  complete_pricing_requests: 2,
  partial_pricing_requests: 0,
  avg_latency_ms: 4600,
};
const model = (name, provider, extra = {}) => ({
  id: name,
  name,
  alias: name + "-alias",
  routable_ids: [name, name + "-alias"],
  provider,
  configured_provider: provider,
  declared_providers: [provider],
  candidate_providers: [provider],
  pool: "",
  desc: "Fixture model",
  locality: provider === "omlx" ? "local" : "cloud",
  tools: null,
  composite: null,
  fallback_model: null,
  vision_route: null,
  provider_model_id: "upstream-" + name,
  context: 262144,
  max_output_tokens: 8192,
  thinking: "optional",
  thinking_levels: ["low", "high"],
  thinking_format: "",
  vision: true,
  pricing: provider === "omlx" ? null : { input: 1, output: 2 },
  pricing_status: provider === "omlx" ? "unmetered" : "metered",
  enabled: true,
  available: true,
  provider_ready: true,
  provider_configured: true,
  ...extra,
});
const provider = (id, extra = {}) => ({
  id,
  configured: true,
  enabled_models: 1,
  base_url: "https://" + id + ".example/v1",
  protocol: "openai",
  has_api_key: true,
  ready: true,
  issues: [],
  pool_memberships: [],
  ...extra,
});
const recent = {
  requests: 10,
  failures: 1,
  rate_limits: 0,
  last_status: 200,
  last_request_at: now,
};
const member = (id, position, ready) => ({
  id,
  position,
  role: position === 1 ? "primary" : "secondary",
  configured: true,
  ready,
  active: ready,
  base_url: "https://" + id + ".cloud.databricks.com",
  auth_refresh: "databricks-cli",
  auth_profile: id + "-profile",
  credential_type: "oauth",
  token_state: ready ? "valid" : "expired",
  endpoint_style: "invocations",
  issues: ready ? [] : ["circuit_open"],
  circuit: {
    is_open: !ready,
    consecutive_failures: ready ? 0 : 3,
    last_failure_status: ready ? 0 : 503,
  },
  recent,
});
function fixture() {
  const members = [
    member("ws-primary", 1, false),
    member("ws-backup", 2, true),
  ];
  return {
    status: {
      status: "ok",
      uptime_seconds: 12345,
      pid: 999,
      writes_enabled: true,
      config_path: "/fixture/config.yaml",
      model_info_path: "/fixture/models.json",
      auth: { client_auth_enabled: true, admin_auth_enabled: true },
    },
    providers: [
      provider("omlx", { base_url: "http://localhost:9110/v1" }),
      provider("cloud"),
      provider("ws-primary", {
        pool_memberships: [{ pool: "claude-routing", position: 1 }],
      }),
      provider("ws-backup", {
        enabled_models: 0,
        pool_memberships: [{ pool: "claude-routing", position: 2 }],
      }),
      provider("unused", { enabled_models: 0 }),
    ],
    models: [
      model("local-model", "omlx", { tools: true }),
      model("cloud-model", "cloud", {
        vision: false,
        vision_route: { model: "local-model", mode: "extract_then_answer" },
      }),
      model("pooled-model", "ws-primary", {
        pool: "claude-routing",
        declared_providers: ["ws-primary", "ws-backup"],
        candidate_providers: ["ws-primary", "ws-backup"],
      }),
    ],
    pools: [
      {
        id: "claude-routing",
        state: "degraded",
        active_member: "ws-backup",
        ready_members: 1,
        models: ["pooled-model"],
        members,
      },
    ],
    requests: [
      {
        id: 1,
        ts: now,
        ts_iso: "2026-01-01T12:00:00",
        model: "local-model",
        provider: "omlx",
        provider_model_id: "upstream-local-model",
        endpoint: "/v1/chat/completions",
        status: 200,
        error: null,
        latency_ms: 4600,
        cost_usd: 0,
        pricing_complete: 1,
        usage_reported: 1,
        is_stream: 0,
        input_tokens: 300,
        output_tokens: 50,
      },
      {
        id: 2,
        ts: now - 20,
        model: "pooled-model",
        provider: "ws-primary",
        status: 200,
        error: "Stream disconnected",
        latency_ms: 74000,
        cost_usd: null,
        pricing_complete: 0,
        is_stream: 1,
      },
      {
        id: 3,
        ts: now - 40,
        model: "cloud-model",
        provider: "cloud",
        status: 200,
        error: null,
        latency_ms: 1200,
        cost_usd: 0.032,
        pricing_complete: 1,
      },
    ],
  };
}
let state = fixture(),
  fail = new Set(),
  delays = new Map(),
  writes = [],
  invalidAuth = false;
const usage = () => ({
  summary,
  by_route: state.models.map((m) => ({
    ...summary,
    dim: m.name,
    provider: m.provider,
    provider_model_id: m.provider_model_id,
    route_complete: 1,
  })),
  by_provider: state.providers
    .slice(0, 2)
    .map((p) => ({ ...summary, dim: p.id })),
});
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");
  const pathname = url.pathname.replace(/^\/gateway/, "");
  const respond = (code, data) => {
    res.writeHead(code, { "Content-Type": "application/json" });
    res.end(JSON.stringify(data));
  };
  if (pathname === "/admin") {
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end(html);
    return;
  }
  if (pathname === "/health") {
    respond(200, { ok: true });
    return;
  }
  if (!pathname.startsWith("/admin/api/")) {
    respond(404, {});
    return;
  }
  if (req.headers.authorization !== "Bearer fixture-admin" || invalidAuth) {
    respond(401, { detail: "Invalid key" });
    return;
  }
  if (delays.has(pathname))
    await new Promise((resolve) => setTimeout(resolve, delays.get(pathname)));
  if (fail.has(pathname)) {
    respond(503, { error: { message: "Fixture temporarily unavailable" } });
    return;
  }
  let raw = "";
  for await (const chunk of req) raw += chunk;
  const body = raw ? JSON.parse(raw) : undefined;
  if (req.method !== "GET") {
    writes.push({ path: pathname, method: req.method, body });
    if (!state.status.writes_enabled) {
      respond(403, { detail: "Read-only fixture" });
      return;
    }
    if (pathname.endsWith("/validate")) {
      respond(200, { ok: true, status_code: 200, model_count: 3 });
      return;
    }
    if (pathname.endsWith("/discover")) {
      respond(200, {
        status: "verified",
        models: [{ id: "new-upstream", registered: false }],
      });
      return;
    }
    if (pathname.endsWith("/preview")) {
      respond(200, {
        ok: true,
        routable: true,
        route: body.pool
          ? state.pools[0].members.map((m) => ({
              provider: m.id,
              usable: true,
            }))
          : [{ provider: body.provider, usable: true }],
        issues: [],
        clashes: [],
      });
      return;
    }
    if (pathname.startsWith("/admin/api/providers/")) {
      const id = decodeURIComponent(pathname.split("/").at(-1));
      const found = state.providers.find((p) => p.id === id);
      if (req.method === "DELETE")
        state.providers = state.providers.filter((p) => p.id !== id);
      else if (found)
        Object.assign(found, {
          base_url: body.base_url,
          protocol: body.protocol,
        });
      else state.providers.push(provider(id, body));
      respond(200, { id, reloaded: true });
      return;
    }
    if (pathname.startsWith("/admin/api/models/")) {
      respond(200, { name: pathname.split("/").at(-1), reloaded: true });
      return;
    }
  }
  if (pathname === "/admin/api/status") {
    respond(200, state.status);
    return;
  }
  if (pathname === "/admin/api/providers") {
    respond(200, { providers: state.providers });
    return;
  }
  if (pathname === "/admin/api/models") {
    respond(200, { models: state.models });
    return;
  }
  if (pathname === "/admin/api/config/validation") {
    respond(200, { ok: true, issues: [] });
    return;
  }
  if (pathname === "/admin/api/presets") {
    respond(200, { auto_models: {}, model_presets: {}, routing_profiles: {} });
    return;
  }
  if (pathname === "/admin/api/workspace-pools") {
    respond(200, {
      pools: state.pools,
      standalone_providers: state.providers
        .filter((p) => !p.pool_memberships.length)
        .map((p) => ({ ...p, recent })),
      summary: { pools: state.pools.length },
    });
    return;
  }
  if (pathname === "/admin/api/requests") {
    respond(200, { requests: state.requests });
    return;
  }
  if (pathname === "/admin/api/usage") {
    respond(200, usage());
    return;
  }
  const match = pathname.match(
    /^\/admin\/api\/(models|providers)\/([^/]+)\/stats$/,
  );
  if (match) {
    const id = decodeURIComponent(match[2]),
      kind = match[1] === "models" ? "model" : "provider";
    respond(200, {
      [kind]: state[match[1]].find((m) => (m.name || m.id) === id) || null,
      routable_ids: [id, id + "-alias"],
      usage: summary,
      recent: state.requests,
    });
    return;
  }
  const requestMatch = pathname.match(/^\/admin\/api\/requests\/(\d+)$/);
  if (requestMatch) {
    respond(200, {
      request: state.requests.find((r) => r.id === Number(requestMatch[1])),
    });
    return;
  }
  respond(404, {});
});
await new Promise((resolve) =>
  server.listen(
    process.argv.includes("--serve") ? 9127 : 0,
    "127.0.0.1",
    resolve,
  ),
);
const base = "http://127.0.0.1:" + server.address().port;
if (process.argv.includes("--serve")) {
  console.log(
    "Isolated admin UI fixture: " +
      base +
      "/admin (key: fixture-admin, no production data)",
  );
  process.on("SIGTERM", () => server.close(() => process.exit(0)));
} else {
  let browser;
  try {
    const { chromium } = require("playwright");
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        viewport: { width: 1440, height: 1000 },
      }),
      page = await context.newPage(),
      errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const click = async (selector) => {
      await page.locator(selector).first().click();
      if (selector.includes("data-open-"))
        await page.waitForFunction(
          () => !document.querySelector("#drawerBody .loading-bar"),
        );
      if (/^#(save|preview|validate|discover|delete)/.test(selector))
        await page
          .locator(selector)
          .waitFor({ state: "visible" })
          .then(() =>
            page.waitForFunction(
              (id) => !document.querySelector(id).disabled,
              selector,
            ),
          );
    };
    const refresh = async () => {
      await click("#refreshBtn");
      await page.waitForFunction(
        () => !document.getElementById("refreshBtn").disabled,
      );
    };
    const nav = async (tab) => {
      await click('[data-tab="' + tab + '"]');
    };
    await page.goto(base + "/admin");
    await page.fill("#adminKey", "fixture-admin");
    await click("#unlockBtn");
    await page.waitForFunction(
      () =>
        !document.getElementById("refreshBtn").disabled &&
        document.getElementById("modelsMeta").textContent === "3 of 3 models",
    );
    assert(await page.locator("#panel-overview").isVisible());
    assert.equal(
      await page.locator(".nav [aria-selected=true]").innerText(),
      "Overview",
    );
    assert.equal(
      await page
        .locator(".stat .detail")
        .first()
        .evaluate((el) => getComputedStyle(el).paddingTop),
      "0px",
    );
    await nav("connections");
    assert.equal(await page.locator("#providers tbody tr").count(), 5);
    await click("#routingGroups > summary");
    assert(
      await page
        .getByText("Primary skipped; backup preferred", { exact: true })
        .isVisible(),
    );
    assert.equal(
      await page.locator("#workspaceList .workspace-row").count(),
      2,
    );
    assert.equal(
      await page
        .locator('[data-copy-command="model-gateway workspace repair"]')
        .count(),
      1,
    );
    await click('#providers [data-open-provider="ws-backup"]');
    assert.match(await page.locator("#drawerBody").innerText(), /pooled-model/);
    await click("[data-close-detail]");
    await nav("models");
    await page.fill("#modelFilter", "zz-no-match");
    assert.match(
      await page.locator("#models tbody").innerText(),
      /No models match/,
    );
    await page.fill("#modelFilter", "");
    await page.selectOption("#modelCapability", "tools");
    assert.equal(await page.locator("#models tbody tr").count(), 1);
    await page.selectOption("#modelCapability", "");
    await click('#models [data-open-model="cloud-model"]');
    assert.match(
      await page.locator("#drawerBody").innerText(),
      /this model remains the answerer/,
    );
    assert(await page.locator("#appChrome").evaluate((el) => el.inert));
    await click('#drawer [data-open-provider="cloud"]');
    assert.equal(
      await page.locator(":focus").getAttribute("id"),
      "drawer",
      "Nested detail transition lost focus",
    );
    await page.keyboard.press("Escape");
    assert(!(await page.locator("#appChrome").evaluate((el) => el.inert)));
    assert.equal(
      await page.locator(":focus").getAttribute("id"),
      "tab-connections",
      "Focus returned to a hidden model row",
    );
    await nav("models");
    await click('#models [data-open-model="pooled-model"]');
    await click('[data-edit-model="pooled-model"]');
    assert.equal(await page.inputValue("#mPool"), "claude-routing");
    assert(await page.locator("#mPool").isDisabled());
    await click("#previewModelBtn");
    await page.waitForFunction(() =>
      document.getElementById("mMsg").textContent.includes("Preview complete"),
    );
    assert.equal(writes.at(-1).body.pool, "claude-routing");
    await click("#saveModelBtn");
    await page.waitForFunction(() =>
      document
        .getElementById("mMsg")
        .textContent.includes("Saved and activated"),
    );
    assert(!Object.hasOwn(writes.at(-1).body, "pool"));
    await click("[data-cancel-model]");
    await nav("connections");
    await click('#providers [data-open-provider="cloud"]');
    await click('[data-edit-provider="cloud"]');
    await page.fill("#pApiKey", "unsaved-private-key");
    await page.fill("#pBaseUrl", "https://unsaved.invalid");
    await click('#providers [data-open-provider="omlx"]');
    await click('[data-edit-provider="omlx"]');
    assert.equal(await page.inputValue("#pApiKey"), "");
    assert.equal(
      await page.inputValue("#pBaseUrl"),
      "http://localhost:9110/v1",
    );
    await click("#saveProviderBtn");
    await page.waitForFunction(() =>
      document
        .getElementById("pMsg")
        .textContent.includes("Saved and activated"),
    );
    assert.equal(writes.at(-1).body.base_url, "http://localhost:9110/v1");
    assert(!Object.hasOwn(writes.at(-1).body, "api_key"));
    await click("#validateProviderBtn");
    await page.waitForFunction(() =>
      document
        .getElementById("pMsg")
        .textContent.includes("Inference was not tested"),
    );
    assert.match(
      await page.locator("#providers").innerText(),
      /Inventory accessible/,
    );
    await click("#discoverBtn");
    await page.waitForSelector("#discoverPanel:not(.hidden)");
    await click("[data-register-discovered]");
    assert.equal(await page.inputValue("#mContext"), "");
    assert.equal(await page.inputValue("#mPricing"), "");
    assert.equal(await page.inputValue("#mDesc"), "");
    await click("[data-cancel-model]");
    await nav("activity");
    await page.selectOption("#requestOutcome", "failed");
    assert.equal(await page.locator("#recentReq tbody tr").count(), 1);
    await click("#recentReq [data-open-request]");
    assert.match(
      await page.locator("#drawerBody").innerText(),
      /Failover \/ attempt chain[\s\S]*Not recorded/,
    );
    await click("[data-close-detail]");
    await page.selectOption("#requestOutcome", "");
    await click('[data-activity="usage"]');
    await click('[data-w="7d"]');
    await page.waitForFunction(() =>
      document
        .getElementById("usageRange")
        .textContent.startsWith("Last 7 days ·"),
    );
    assert.match(await page.locator("#usageView").innerText(), /4.6 s/);
    for (const [hash, tab, detail] of [
      ["#providers/cloud", "connections", true],
      ["#pools", "connections", false],
      ["#presets", "models", false],
      ["#usage/1", "activity", true],
      ["#debug", "settings", false],
    ]) {
      await page.goto(base + "/admin" + hash);
      await page.waitForFunction(
        () =>
          !document.getElementById("refreshBtn").disabled &&
          !document.getElementById("dash").classList.contains("hidden"),
      );
      assert(await page.locator("#panel-" + tab).isVisible());
      if (detail) {
        await page.waitForSelector("#drawer:not([hidden])");
        await click("[data-close-detail]");
      }
    }
    await page.goto(base + "/admin");
    await page.waitForFunction(
      () => !document.getElementById("refreshBtn").disabled,
    );
    await nav("models");
    await nav("connections");
    await page.goBack();
    assert(await page.locator("#panel-models").isVisible());
    await nav("overview");
    fail.add("/admin/api/workspace-pools");
    await refresh();
    assert(await page.locator("#dashErr").isVisible());
    assert.match(await page.locator("#dashErr").innerText(), /stale/);
    assert(await page.locator("#panel-overview").isVisible());
    fail.clear();
    await refresh();
    assert(!(await page.locator("#dashErr").isVisible()));
    await nav("connections");
    delays.set("/admin/api/providers/cloud/stats", 150);
    delays.set("/admin/api/providers/omlx/stats", 400);
    fail.add("/admin/api/providers/cloud/stats");
    await page.locator('#providers [data-open-provider="cloud"]').click();
    await page.evaluate(() =>
      document.querySelector('#providers [data-open-provider="omlx"]').click(),
    );
    await page.waitForTimeout(500);
    assert.match(await page.locator("#drawerBody h2").innerText(), /omlx/);
    assert(
      !(await page.locator("#drawerBody").innerText()).includes(
        "Fixture temporarily unavailable",
      ),
    );
    delays.clear();
    fail.clear();
    await click("[data-close-detail]");
    const labels = await page
      .locator("input,select,textarea")
      .evaluateAll((els) =>
        els
          .filter((el) => !el.labels?.length && !el.getAttribute("aria-label"))
          .map((el) => el.id),
      );
    assert.deepEqual(labels, []);
    await page.focus("#tab-overview");
    await page.keyboard.press("ArrowRight");
    assert.equal(await page.locator(":focus").getAttribute("id"), "tab-models");
    const screenshotDir = process.env.ADMIN_UI_SCREENSHOT_DIR;
    if (screenshotDir) mkdirSync(screenshotDir, { recursive: true });
    for (const viewport of [
      { width: 390, height: 844 },
      { width: 820, height: 1180 },
      { width: 1440, height: 1000 },
    ]) {
      await page.setViewportSize(viewport);
      for (const tab of [
        "overview",
        "models",
        "connections",
        "activity",
        "settings",
      ]) {
        await nav(tab);
        assert(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
          `${tab} overflows at ${viewport.width}`,
        );
        if (screenshotDir)
          await page.screenshot({
            path: path.join(screenshotDir, `${viewport.width}-${tab}.png`),
            fullPage: true,
          });
      }
      await nav("models");
      await click('#models [data-open-model="local-model"]');
      assert(
        await page.evaluate(
          () => document.getElementById("drawer").scrollWidth <= innerWidth,
        ),
        "Detail content overflows",
      );
      await page.focus("#drawer");
      await page.keyboard.press("Tab");
      assert(
        await page.locator(":focus").evaluate((el) => !!el.closest("#drawer")),
        "Focus escaped detail view",
      );
      await page.keyboard.press("Shift+Tab");
      assert(
        await page.locator(":focus").evaluate((el) => !!el.closest("#drawer")),
        "Reverse focus escaped detail view",
      );
      await page.locator("#drawer").evaluate((el) => {
        el.scrollTop = 0;
      });
      if (screenshotDir)
        await page.screenshot({
          path: path.join(screenshotDir, `${viewport.width}-model-detail.png`),
          fullPage: false,
        });
      await click("[data-close-detail]");
    }
    await page.emulateMedia({ colorScheme: "dark" });
    await nav("overview");
    if (screenshotDir)
      await page.screenshot({
        path: path.join(screenshotDir, "1440-overview-dark.png"),
        fullPage: true,
      });
    await page.emulateMedia({ colorScheme: "light" });
    state.status.writes_enabled = false;
    await refresh();
    await nav("connections");
    assert.equal(await page.locator("[data-add-provider]:visible").count(), 0);
    await click('#providers [data-open-provider="cloud"]');
    assert.equal(
      await page.locator("#drawer [data-edit-provider]:visible").count(),
      0,
    );
    await click("[data-close-detail]");
    state.models = [];
    state.providers = [];
    state.pools = [];
    state.requests = [];
    await refresh();
    await nav("overview");
    assert(await page.locator("#overviewSetup").isVisible());
    await nav("models");
    assert.match(
      await page.locator("#models tbody").innerText(),
      /No models registered/,
    );
    await nav("connections");
    assert.match(
      await page.locator("#providers tbody").innerText(),
      /No connections configured/,
    );
    invalidAuth = true;
    await refresh();
    assert(await page.locator("#lockedView").isVisible());
    assert.equal(await page.inputValue("#adminKey"), "");
    assert.equal(
      await page.evaluate(() => sessionStorage.getItem("mg-admin-key")),
      null,
    );
    invalidAuth = false;
    state = fixture();
    await page.goto(base + "/gateway/admin#models/local-model");
    await page.fill("#adminKey", "fixture-admin");
    await click("#unlockBtn");
    await page.waitForSelector("#drawer:not([hidden])");
    await page.waitForFunction(() =>
      document
        .getElementById("drawerBody")
        .textContent.includes("Copy request example"),
    );
    assert.match(
      await page.locator("#drawerBody pre").innerText(),
      /\/gateway\/v1\/chat\/completions/,
    );
    assert.deepEqual(errors, []);
    console.log(
      "PASS: navigation, legacy links, back navigation, pools, scoped commands, filters, details, safe editing, discovery, stale/error/auth states, request accounting, keyboard, labels, responsive layouts, read-only, empty state, mounted base path.",
    );
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}
