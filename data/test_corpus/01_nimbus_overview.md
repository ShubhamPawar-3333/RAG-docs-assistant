# Nimbus Analytics — Product Overview

Nimbus Analytics is a cloud-based **product analytics platform** that helps software
teams understand how people use their applications. It is delivered as a fully managed
SaaS product; there is no self-hosted edition.

## Core capabilities

- **Event tracking** — capture user actions (clicks, page views, custom events) from web
  and mobile apps.
- **Funnels** — measure conversion through a sequence of steps and see where users drop off.
- **Dashboards** — build shareable boards of charts that refresh automatically.
- **Cohort analysis** — group users by behaviour or attributes and compare retention over time.
- **Insight Engine** — an automated anomaly-detection system that flags unexpected spikes or
  drops in any tracked metric and notifies the workspace owner.

## How data gets in

Applications send data to Nimbus through one of the official SDKs or the REST Events API:

| Method            | Platforms                                   |
|-------------------|---------------------------------------------|
| Client SDKs       | JavaScript (browser), iOS, Android           |
| Server SDK        | Python                                        |
| Events API        | Any language, via HTTPS POST                  |

There is **no Ruby or Go SDK** today; server-side customers on those stacks use the Events API directly.

## Known scaling considerations

Large workspaces with more than 50 million monthly events may see slower dashboard load
times. We recommend building **pre-aggregated views** to work around these performance
bottlenecks. The query engine parallelises across shards, but very wide date ranges on
raw event tables remain the most common cause of slow queries.

## Terminology

- **Workspace** — an isolated container for one product's data, billing, and members.
- **Project** — a data source inside a workspace (for example, "Web" and "iOS" projects).
- **Seat** — one named member with login access to a workspace.
