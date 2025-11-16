# Msanii AI

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

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📝 License

This project is private and proprietary.

