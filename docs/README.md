# Developer Documentation

Domain-specific guides organized by tool/product area. Each doc covers backend endpoints, frontend hooks/components, database tables, and how to test locally.

## Tool Guides

| Doc | Tool | Route | Covers |
|-----|------|-------|--------|
| [Workspace](workspace.md) | Workspace | `/workspace` | Integrations (Drive, Slack), Kanban boards, Calendar, Notifications, Settings |
| [Portfolio](portfolio.md) | Portfolio & Projects | `/portfolio`, `/projects/:id` | Artists, projects, files, audio, members, notes |
| [Registry](registry.md) | Metadata Registry | `/tools/registry` | Works, ownership stakes, collaborators, licensing, agreements |
| [OneClick](oneclick.md) | OneClick | `/tools/oneclick` | Royalty calculator, PDF generation, sharing to Drive/Slack |
| [Zoe](zoe.md) | Zoe AI | `/tools/zoe` | AI contract analysis chatbot, streaming, document processing |
| [Split Sheet](split-sheet.md) | Split Sheet | `/tools/split-sheet` | Split sheet PDF/DOCX generator |

## Operational Guides

| Doc | Covers |
|-----|--------|
| [Secrets & Env Vars](secrets.md) | Every env var the app reads, where to get it, what's required to boot vs feature-optional, prod-vs-local storage |
| [Stripe Integration](stripe-integration.md) | Checkout + Portal + webhook wiring, full local-test flow with Stripe CLI, prod cutover checklist |
| [Admin Roles](admin-roles.md) | `ADMIN_EMAILS` bootstrap + `profiles.is_admin` DB grants, `/admin/users` console, promote/demote flow, three-layer gating |

## Other Documentation

- **[README.md](../README.md)** — Setup, installation, deployment
- **[CLAUDE.md](../.claude/CLAUDE.md)** — Architecture overview, conventions, coding guidelines
- **[.env.example](../.env.example)** — All environment variables with setup comments
