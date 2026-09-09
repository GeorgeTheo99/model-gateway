# Admin UI redesign: approved design brief

Status: design approved; local implementation subsequently authorized by the user.
Deployment and Git pushes require separate approval.

## Purpose and scope

Make Model Gateway understandable through operator tasks rather than its internal
configuration structure. The operator should quickly understand what is available,
how to use a model, where requests will route, and what needs attention.

The user approved a broader, provider-neutral redesign. Databricks is not the
first implementation priority, but users choosing Databricks must receive clear,
complete workspace-pool visibility in the relevant workflow.

This is a written, screen-by-screen UX proposal, not an implemented UI or visual
mockup. Retain the calm, restrained product style described in `PRODUCT.md`.
Preserve the embedded, no-build UI architecture initially. Visual mockups are
outside this planning scope.

## Primary operator questions

1. What needs attention?
2. Which model should I call, and how?
3. Where will that request go?
4. What happened when it failed?

## Navigation

Four primary destinations:

- **Overview**: health, attention, and recent activity.
- **Models**: find, understand, and use models.
- **Connections**: manage and understand providers, runtimes, and routing groups.
- **Activity**: investigate requests and understand usage/cost.

Settings and diagnostics are secondary navigation. Databricks-specific concepts
must not occupy the global navigation for users who do not use Databricks.

Existing API contracts remain unchanged. Existing deep links must remain useful,
with compatibility mappings where navigation changes.

## 1. Overview

Default landing page, replacing the current Presets landing view.

- Compact status bar: gateway availability, configuration status, and refresh time.
  Uptime and process details are secondary.
- Needs attention: actionable issues with affected models and consequences, not
  unexplained badges. Explain what happened, what is affected, and what to do.
- Recent activity: request/error/spend summaries and a short recent-request list.
- Entry points: Find a model, Add a connection, View failures.

A fresh installation offers a first-connection path instead of empty tables.
A failed refresh preserves previous data only with an explicit stale indicator
and a visible error beside the affected section.

## 2. Models

Searchable inventory with a reduced default column set:

**Model | Local/cloud | Capabilities | Connection | Availability**

Filters include local/cloud, provider, vision, and tools. Display readable limits
such as `256K`, preserving exact values in details.

A dedicated model detail view contains:

- **Use this model**: client-facing ID, aliases, gateway URL, and a copyable request
  example without embedded credentials.
- **Capabilities & cost**: tools, vision, reasoning, context/output limits, and
  explicit pricing status.
- **Routing**: direct connection or workspace pool, preference order, and separately
  identified model/vision fallback behavior.
- **Recent activity**: requests, errors, latency, and cost for this model.
- **Configuration**: an explicit edit action with advanced fields progressively
  disclosed.

Upstream IDs are secondary rather than dominant inventory columns. Distinguish
being enabled/configured from being verified working.

Configured gateway-owned presets belong within Models as grouped selections,
not as an empty default landing page. This does not authorize changing profile
permissions or exposing consumer-private configuration through admin APIs.

## 3. Connections

Unified home for cloud providers and local runtimes. Each row answers:

- What is this connection?
- How many models use it?
- Is its configuration complete?
- What do recent requests or an explicit check tell us?
- Does anything require attention?

Separate configuration readiness, last verification, and recent outcomes rather
than compressing them into one green status. Connections without assigned models
are marked **Unused**.

### Databricks detail

Provide richer visibility inside Connections and within affected model routes.

#### Routing groups (workspace pools)

Explain the primitive: **Try these workspaces in order for the same model.**
This is ordered failover, not load balancing.

For each group, show:

1. Assigned models.
2. Workspace preference order.
3. Which workspace would be tried next, and why.
4. Whether usable backups remain.

Distinguish primary available, primary skipped with backup preferred, no eligible
workspace, and upstream availability not verified. Do not equate local routing
readiness with a successful upstream check, or configured model association with
verified endpoint coverage.

#### Workspaces

Show each workspace once, with authentication status, group memberships, affected
models, and recent outcomes. CLI profiles, endpoint styles, and circuit-breaker
details belong under Diagnostics.

#### Recovery

Instructions and actions must state scope and effect.

- Label copied workspace test commands explicitly and explain that tests can make
  real inference requests.
- Put global OAuth-workspace repair at the Databricks level, not inside every pool.
- Do not present browser-based repair or reordering as operational until safe
  backend support exists.

Standalone provider inventory stays in Connections instead of being repeated in
full beneath workspace pools.

## 4. Activity

Two views:

- **Requests**: searchable history with model, actual provider/workspace when
  recorded, outcome, duration, and cost.
- **Usage & cost**: time-window summaries and model/connection breakdowns.

Request detail explains requested versus actual routing, recorded fallback or
failover, timing, and accounting completeness. Missing evidence is **Not recorded**,
not an implied absence of failover or failure.

Use readable durations such as `4.6 s` and `2m 10s`. Expand raw token accounting
and protocol details on demand.

Time scope must be unambiguous. A usage-window selector must not appear to filter
an independent latest-requests list when it does not.

## Shared interaction requirements

- Compact navigation and status chrome; prioritize the current task's content.
- Consistent links between models, connections, and requests.
- Explicit loading, stale, empty, error, and success states.
- Keyboard-accessible tables, correctly associated form labels, and properly
  managed focus for detail views and overlays.
- Editing begins with the selected entity's populated non-secret fields, a cleared
  secret field, and explicit Save/Cancel controls.
- Destructive or activation actions explain scope and affected models.
- Read-only mode is globally visible; unavailable operations explain why.
- Debug and raw configuration live under secondary Settings & diagnostics.

## Review evidence and implementation guardrails

The review verified that local live `/admin` HTML matched the checkout. Actual
local runtime data contained zero workspace pools and eight standalone providers.
Pool behavior was reviewed in source and a clearly labeled synthetic failover
scenario based on `tests/test_management.py`, not a live Databricks outage.

Source and visual review identified these existing issues to address during an
authorized implementation:

- Refresh errors can be hidden inside the Debug panel while old status remains.
- Pool labels overstate verification; the no-probe qualification is buried.
- Provider editing can retain values belonging to a previously selected provider.
- Every per-pool repair button copies the same global repair command.
- Global `.detail` CSS collides with metric annotation spans, adding nested boxes.
- Form-label associations, keyboard drill-downs, tab semantics, and overlay focus
  behavior need correction.

Preserve backend safety gates, secret redaction, and identity boundaries. Existing
readiness signals must not be relabeled as stronger runtime evidence. New request
filters, verification timestamps, guided onboarding, or management actions may
require API work; inspect support before promising functionality.

The first implementation should deliver the shared structure and complete
browse-to-understand-to-diagnose workflows. It need not start with Databricks.

## Acceptance and authorization

The user approved this structure and contextual Databricks workflow as the design
brief, then separately authorized implementation. That implementation authorization
does not authorize live provider probes, production configuration changes,
deployment, or Git pushes.

Success is an operator answering the four primary questions without reconstructing
routing from provider IDs, raw configuration, and scattered diagnostics.
