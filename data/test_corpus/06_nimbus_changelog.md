# Nimbus Analytics — Changelog

All dates are in UTC. Newest releases first.

## v3.3.0 — 2026-09-02

- **Data residency in the EU**: workspaces can now pin all storage to `eu-west-1`.
- Scheduled report emails: send any dashboard as a PDF on a daily or weekly schedule.
- Fixed a bug where deleted funnels still appeared in search results.

## v3.2.0 — 2026-07-14

- **SCIM provisioning** for automated user management (Pro and Enterprise).
- Funnel conversion windows are now configurable per funnel (previously fixed at 7 days).
- Query engine is roughly **20% faster** on cohort queries.

## v3.1.0 — 2026-04-08

- **Insight Engine** anomaly detection reaches general availability.
- The **Python server SDK** is released.
- Added CSV export to the funnels page.

## v3.0.0 — 2026-01-20

- Rebuilt dashboard builder with drag-and-drop tiles.
- Dark mode.
- **Breaking change**: the v1 Events API is removed. All clients must use `/v2`.
