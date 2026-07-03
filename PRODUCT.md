# Product

## Register

product

## Users

A single operator (the machine owner) running a self-hosted model router on a home server. They open the admin UI on a desktop browser, usually on the same LAN, to check that routing is healthy, audit which models are exposed, glance at recent usage/cost, and occasionally add or edit a provider or model. Sessions are short and task-driven: "is it up?", "what did the last request cost?", "did the local oMLX path come back?".

## Product Purpose

Model Gateway is the single facade in front of every model the operator routes to, both cloud providers (Anthropic, OpenAI, Google, etc.) and local oMLX models on this machine. The admin UI exists to give a fast, trustworthy read on that facade: provider and model inventory, config readiness, usage and cost, recent requests, and write-back controls for providers/models. Success is the operator trusting the dashboard at a glance and rarely needing to open a terminal.

## Brand Personality

Calm, precise, dense. The interface should feel like a quiet instrument, not a marketing surface. It earns trust by showing the right thing in the right place and getting out of the way.

## Anti-references

- Generic SaaS dashboards: cream backgrounds, navy/blue accents, big hero metric cards in a row, gradient accents.
- Cloud-provider console clutter: noisy tables, eighteen columns, chrome that buries the signal.
- "AI made that" gradients, glassmorphism, and decorative motion.
- Anything that hides status behind a marketing sheen.

## Design Principles

- **Status first, chrome second.** The eye should land on health and the latest signal before any heading or affordance.
- **Density without noise.** Pack information for scanning, but separate it with space and type, not borders and boxes.
- **Honest empty and error states.** A missing admin key, an empty usage window, or a failing provider is explained in place, never rendered as a broken dashboard.
- **One calm surface.** Restrained palette, one accent for state and primary actions, tinted neutrals throughout. No decorative gradients or glass.
- **Keyboard and at-a-glance friendly.** Admin key entry, refresh, and window switching should be reachable without a mouse; values should be readable without hovering.

## Accessibility & Inclusion

- WCAG AA contrast on all text against its surface, including muted labels and table data.
- Visible focus rings on every interactive control.
- Honors `prefers-reduced-motion` (any transitions stay purely state-conveying and short, and are removed under reduced motion).
- Status conveyed by text and shape, not color alone (pill labels and words accompany color cues).
