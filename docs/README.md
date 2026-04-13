# Developer Documentation

Domain-specific guides organized by tool/product area. Each doc covers backend endpoints, frontend hooks/components, database tables, and how to test locally.

## Tool Guides

| Doc | Tool | Route | Covers |
|-----|------|-------|--------|
| [Workspace](workspace.md) | Workspace | `/workspace` | Integrations (Drive, Slack), Kanban boards, Calendar, Notifications, Settings |
| [Portfolio](portfolio.md) | Portfolio & Projects | `/portfolio`, `/projects/:id` | Artists, projects, files, audio, members, notes |
| [Registry](registry.md) | Rights Registry | `/tools/registry` | Works, ownership stakes, collaborators, licensing, agreements |
| [OneClick](oneclick.md) | OneClick | `/tools/oneclick` | Royalty calculator, PDF generation, sharing to Drive/Slack |
| [Zoe](zoe.md) | Zoe AI | `/tools/zoe` | AI contract analysis chatbot, streaming, document processing |
| [Split Sheet](split-sheet.md) | Split Sheet | `/tools/split-sheet` | Split sheet PDF/DOCX generator |

## Other Documentation

- **[README.md](../README.md)** — Setup, installation, deployment
- **[CLAUDE.md](../.claude/CLAUDE.md)** — Architecture overview, conventions, coding guidelines
- **[.env.example](../.env.example)** — All environment variables with setup comments
