import { lazy, Suspense, useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigationType } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { ThemeProvider } from "@/components/ThemeProvider";
import { ThemeToggle } from "@/components/ThemeToggle";
import { AdminProtectedRoute } from "@/components/AdminProtectedRoute";
import { usePageTimer } from "@/hooks/usePageTimer";
import { useAdminPosthogTag } from "@/hooks/useAdminPosthogTag";
import { TesterBanner } from "@/components/tester/TesterBanner";

// Eager — small, needed on initial load
import Index from "./pages/Index";
import Auth from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import NotFound from "./pages/NotFound";

// Lazy-load — heavy pages loaded on demand
const Artists = lazy(() => import("./pages/Artists"));
const ArtistProfile = lazy(() => import("./pages/ArtistProfile"));
const NewArtist = lazy(() => import("./pages/NewArtist"));
const Tools = lazy(() => import("./pages/Tools"));
const OneClick = lazy(() => import("./pages/OneClick"));
const OneClickDocuments = lazy(() => import("./pages/OneClickDocuments"));
const Zoe = lazy(() => import("./pages/Zoe"));
const Profile = lazy(() => import("./pages/Profile"));
const Workspace = lazy(() => import("./pages/Workspace"));
const Notifications = lazy(() => import("./pages/Notifications"));
const WorkspaceBoards = lazy(() => import("./pages/WorkspaceBoards"));
const Portfolio = lazy(() => import("./pages/Portfolio"));
const SplitSheet = lazy(() => import("./pages/SplitSheet"));
const Onboarding = lazy(() => import("./pages/Onboarding"));
const Documentation = lazy(() => import("./pages/Documentation"));
const Registry = lazy(() => import("./pages/Registry"));
const ExpenseTracker = lazy(() => import("./pages/ExpenseTracker"));
const WorkDetail = lazy(() => import("./pages/WorkDetail"));
const InviteClaim = lazy(() => import("./pages/InviteClaim"));
const ProjectDetail = lazy(() => import("./pages/ProjectDetail"));
const ConfirmEmail = lazy(() => import("./pages/ConfirmEmail"));
const Pricing = lazy(() => import("./pages/Pricing"));
const AdminUsers = lazy(() => import("./pages/AdminUsers"));
const Organization = lazy(() => import("./pages/Organization"));
const OrgInviteClaim = lazy(() => import("./pages/OrgInviteClaim"));
const Team = lazy(() => import("./pages/Team"));
const About = lazy(() => import("./pages/About"));
const Features = lazy(() => import("./pages/Features"));
const Privacy = lazy(() => import("./pages/Privacy"));
const Security = lazy(() => import("./pages/Security"));
const Contact = lazy(() => import("./pages/Contact"));

const queryClient = new QueryClient();

const PageLoader = () => (
  <div className="min-h-screen bg-background flex items-center justify-center">
    <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
  </div>
);

function PageTimer() {
  usePageTimer();
  return null;
}

// /subscription merged into /profile (Account & Billing). The redirect keeps
// the query string so Stripe success URLs minted before the merge
// (?stripe_session_id=...&welcome=true) still reach the post-checkout handler.
function SubscriptionRedirect() {
  const { search } = useLocation();
  return <Navigate to={`/profile${search}`} replace />;
}

// /organization renamed to /teams (2026-08-16 teams rebrand); same shape.
function TeamsRedirect() {
  const { search } = useLocation();
  return <Navigate to={`/teams${search}`} replace />;
}

/**
 * Reset scroll on navigation. React Router keeps the current scroll offset
 * across route changes, so following a link from a page footer (e.g. Contact)
 * lands you partway down the new page.
 *
 * Skipped for POP so the browser's own scroll restoration still works on
 * back/forward. Keyed on pathname only — same-path `?query` changes (the
 * `/docs?section=` links) are handled by the page itself.
 */
function ScrollToTop() {
  const { pathname } = useLocation();
  const navigationType = useNavigationType();
  useEffect(() => {
    if (navigationType === "POP") return;
    window.scrollTo({ top: 0, left: 0 });
  }, [pathname, navigationType]);
  return null;
}

function AdminPosthogTagger() {
  useAdminPosthogTag();
  return null;
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <ThemeToggle />
        <BrowserRouter>
          <ScrollToTop />
          <PageTimer />
          <AuthProvider>
            <TesterBanner />
            <AdminPosthogTagger />
            <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/auth" element={<Auth />} />
            <Route path="/auth/confirm-email" element={<ConfirmEmail />} />
            <Route
              path="/onboarding"
              element={
                <ProtectedRoute skipOnboardingCheck>
                  <Onboarding />
                </ProtectedRoute>
              }
            />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/artists"
              element={
                <ProtectedRoute>
                  <Artists />
                </ProtectedRoute>
              }
            />
            <Route
              path="/artists/new"
              element={
                <ProtectedRoute>
                  <NewArtist />
                </ProtectedRoute>
              }
            />
            <Route
              path="/artists/:id"
              element={
                <ProtectedRoute>
                  <ArtistProfile />
                </ProtectedRoute>
              }
            />
            <Route
              path="/tools"
              element={
                <ProtectedRoute>
                  <Tools />
                </ProtectedRoute>
              }
            />
            <Route
              path="/tools/oneclick"
              element={
                <ProtectedRoute>
                  <OneClick />
                </ProtectedRoute>
              }
            />
            <Route
              path="/tools/zoe"
              element={
                <ProtectedRoute>
                  <Zoe />
                </ProtectedRoute>
              }
            />
            <Route
              path="/oneclick/:artistId/documents"
              element={
                <ProtectedRoute>
                  <OneClickDocuments />
                </ProtectedRoute>
              }
            />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <Profile />
                </ProtectedRoute>
              }
            />
            <Route path="/subscription" element={<SubscriptionRedirect />} />
            <Route
              path="/teams"
              element={
                <ProtectedRoute>
                  <Organization />
                </ProtectedRoute>
              }
            />
            {/* Legacy path: invite emails, Stripe return URLs and bookmarks from
                before the teams rename. Redirect keeps the query string
                (?new=1, ?topup=success). */}
            <Route path="/organization" element={<TeamsRedirect />} />
            {/* Public: invited members may not be signed in yet — the page
                shows a sign-in gate and only calls the (auth-required) accept/
                decline endpoints once a user session exists. */}
            <Route path="/orgs/invite/:token" element={<OrgInviteClaim />} />
            <Route
              path="/workspace"
              element={
                <ProtectedRoute>
                  <Workspace />
                </ProtectedRoute>
              }
            />
            <Route
              path="/notifications"
              element={
                <ProtectedRoute>
                  <Notifications />
                </ProtectedRoute>
              }
            />
            <Route
              path="/workspace/boards"
              element={
                <ProtectedRoute>
                  <WorkspaceBoards />
                </ProtectedRoute>
              }
            />
            <Route
              path="/workspace/boards/:artistId"
              element={
                <ProtectedRoute>
                  <WorkspaceBoards />
                </ProtectedRoute>
              }
            />
            <Route
              path="/portfolio"
              element={
                <ProtectedRoute>
                  <Portfolio />
                </ProtectedRoute>
              }
            />
            <Route
              path="/tools/split-sheet"
              element={
                <ProtectedRoute>
                  <SplitSheet />
                </ProtectedRoute>
              }
            />
            <Route path="/docs" element={<Documentation />} />
            <Route path="/pricing" element={<Pricing />} />
            <Route path="/team" element={<Team />} />
            <Route path="/about" element={<About />} />
            <Route path="/features" element={<Features />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/security" element={<Security />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/tools/registry" element={<ProtectedRoute><Registry /></ProtectedRoute>} />
            <Route path="/tools/expense-tracker" element={<ProtectedRoute><ExpenseTracker /></ProtectedRoute>} />
            {/* Public: invited collaborators may not be signed in yet — the page
                shows a sign-in gate and only fetches the (auth-required) preview
                once a user session exists. */}
            <Route path="/tools/registry/invite/:token" element={<InviteClaim />} />
            <Route path="/tools/registry/:workId" element={<ProtectedRoute><WorkDetail /></ProtectedRoute>} />
            <Route path="/projects/:projectId" element={<ProtectedRoute><ProjectDetail /></ProtectedRoute>} />
            <Route
              path="/admin/users"
              element={
                <ProtectedRoute>
                  <AdminProtectedRoute>
                    <AdminUsers />
                  </AdminProtectedRoute>
                </ProtectedRoute>
              }
            />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
            </Suspense>
          </AuthProvider>
        </BrowserRouter>
      </TooltipProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;

