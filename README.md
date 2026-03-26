# Msanii

A modern web application for managing artists and royalties, built with React, TypeScript, and Supabase.

Msanii is a comprehensive platform that helps you organize and track your artist roster, manage royalty information, and streamline your music business operations. Features include secure authentication with Google sign-in, artist profiles, and protected user data.

## 🚀 Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- [Node.js](https://nodejs.org/) 18+ (will be installed via Conda)

### 1. Set Up Conda Environment

Create and activate a new Conda environment:

```bash
# Create a new conda environment with Node.js
conda create -n msanii-ai nodejs -c conda-forge

# Activate the environment
conda activate msanii-ai
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your Supabase credentials (see [Authentication Setup](#authentication-setup) below).

### 4. Run the Project

Start the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## 🔐 Authentication Setup

This app uses Supabase for authentication with Google OAuth support. To set up:

1. Go to [Supabase](https://supabase.com)
2. Access [OneClick](https://supabase.com/dashboard/project/sfugklkakdflrqhmkfps) project
3. Get project credentials from Settings → API
4. Add them to your local `.env` file:
   ```env
   VITE_SUPABASE_URL=your-project-url
   VITE_SUPABASE_ANON_KEY=your-anon-key
   ```

## 📁 Project Structure

```
oneclick/
├── src/
│   ├── components/     # Reusable UI components
│   ├── contexts/       # React context providers
│   ├── hooks/          # Custom React hooks
│   ├── lib/            # Utilities and configurations
│   ├── pages/          # Page components
│   └── App.tsx         # Main app component
├── public/             # Static assets
└── .env               # Environment variables (not in git)
```

## 🛠️ Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## 🔒 Protected Routes

The following routes require authentication:
- `/dashboard` - Main dashboard
- `/artists` - Artist list
- `/artists/new` - Create new artist
- `/artists/:id` - Artist profile
- `/tools` - Tools page
- `/profile` - User profile

## 🌟 Features

- ✅ Google OAuth authentication
- ✅ Email/password authentication
- ✅ Protected routes
- ✅ Artist management
- ✅ User profiles
- ✅ Responsive design with Tailwind CSS
- ✅ Modern UI components with shadcn/ui

## 🚢 Deployment

### Environments

| Environment | Frontend | Backend | Trigger |
|-------------|----------|---------|---------|
| **Dev** | Vercel (auto-deploy from `main`) | Cloud Run (`msanii-backend-dev`) | Push to `main` |
| **Prod** | Vercel (CLI deploy) | Cloud Run (`msanii-backend`) | Published tag release (`v*`) |

Both environments share the same Supabase database — data is user-scoped so there is no cross-contamination.

### Deploy to Dev

Push or merge to `main`. Everything is automatic:
- Frontend: Vercel auto-deploys from `main`
- Backend: GitHub Actions builds and deploys `msanii-backend-dev` to Cloud Run

```bash
git checkout main
git merge your-feature-branch
git push origin main
```

### Deploy to Prod

Create and publish a tag release. Both prod workflows trigger automatically:

```bash
git checkout main
git tag v1.0.0
git push origin v1.0.0
```
This can also be done via GitHub desktop while on main.

Recommended:
Create a release through GitHub: Releases → Draft a new release → Choose tag → Publish.


This triggers:
- `deploy-backend.yml` → deploys `msanii-backend` to Cloud Run
- `deploy-frontend-prod.yml` → deploys to prod Vercel via CLI

### Workflow Files

| Workflow | File | Trigger |
|----------|------|---------|
| Dev backend | `.github/workflows/deploy-backend-dev.yml` | Push to `main` (paths: `src/backend/**`) |
| Prod backend | `.github/workflows/deploy-backend.yml` | Tag push `v*` |
| Prod frontend | `.github/workflows/deploy-frontend-prod.yml` | Tag push `v*` |

### Updating Vercel Environment Variables

The frontend connects to the backend via the `VITE_BACKEND_API_URL` environment variable. After deploying a new Cloud Run service or if the backend URL changes, you need to update this in Vercel:

1. **Get the dev Cloud Run URL:**
   ```bash
   gcloud run services describe msanii-backend-dev --region=northamerica-northeast2 --format='value(status.url)'
   ```

2. **Set the variable in Vercel:**
   - Go to the [Vercel Dashboard](https://vercel.com) → your project → **Settings** → **Environment Variables**
   - Add or update `VITE_BACKEND_API_URL` with the Cloud Run URL (e.g. `https://msanii-backend-dev-xxx-nn.a.run.app`)
   - Scope it to **Preview** and **Development** for the dev backend, or **Production** for the prod backend

3. **Update CORS (if the Vercel URL changed):**
   - Go to GitHub → repo **Settings** → **Secrets and variables** → **Actions**
   - Update `DEV_ALLOWED_ORIGINS` to include the new Vercel dev URL
   - Re-run the backend deploy workflow so Cloud Run picks up the updated CORS origins

4. **Redeploy the frontend** to pick up the new environment variable — either push a new commit to `main` or trigger a redeploy from the Vercel dashboard.

### Required GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `GCP_PROJECT_ID` | GCP project for Cloud Run |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | GCP auth (WIF) |
| `GCP_SERVICE_ACCOUNT_EMAIL` | GCP auth (WIF) |
| `DEV_ALLOWED_ORIGINS` | Dev Vercel URL for CORS |
| `PROD_ALLOWED_ORIGINS` | Prod Vercel URL for CORS |
| `VERCEL_PROD_TOKEN` | Vercel API token for prod deploys |
| `VERCEL_ORG_ID` | Vercel org ID |
| `VERCEL_PROJECT_ID` | Vercel prod project ID |

## 🤝 Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Test thoroughly
4. Submit a pull request to `main`
5. Once merged, changes auto-deploy to dev
6. When ready for prod, create a tag release

## 📝 License

This project is private and proprietary.

