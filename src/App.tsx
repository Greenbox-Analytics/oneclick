import { lazy, Suspense, useEffect, useRef } from "react";
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

// Query-preserving redirect: /subscription → /profile (Stripe success URLs
// minted before the merge carry ?stripe_session_id=...&welcome=true) and
// /organization → /teams (2026-08-16 rebrand; old invite emails).
function RedirectKeepSearch({ to }: { to: string }) {
  const { search } = useLocation();
  return <Navigate to={to + search} replace />;
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

/**
 * Top up the browser's own scroll restoration on back/forward.
 *
 * The native restore fires before React Query has data, so a long list
 * (Portfolio, Registry) is still one screen tall at that moment and the
 * browser can only scroll to the top. This remembers the offset per history
 * entry and re-applies it for up to a second, as the list grows into place.
 *
 * Deliberately additive — `history.scrollRestoration` is left on `auto`, so
 * this only fixes the late-content case and never replaces native behavior
 * (which still handles a plain refresh, where the entry key is new to us).
 * Any real scroll input from the user aborts it immediately: fighting someone
 * who has already started scrolling is worse than landing at the top.
 */
function ScrollRestoration() {
  const { key } = useLocation();
  const navigationType = useNavigationType();

  // Track the offset live in a ref, but only persist it once, on the way out:
  // one storage write per navigation instead of one per scroll event. Reading
  // `window.scrollY` in the cleanup instead would be too late — by the time
  // passive effects run, the next page is already committed and the browser
  // may have clamped the offset to the new (shorter) document.
  const offset = useRef(0);
  useEffect(() => {
    offset.current = 0;
    const onScroll = () => {
      offset.current = window.scrollY;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      try {
        sessionStorage.setItem(`scroll:${key}`, String(offset.current));
      } catch {
        /* private mode / storage disabled — restoration is a nicety */
      }
    };
  }, [key]);

  useEffect(() => {
    if (navigationType !== "POP") return;
    let target: number;
    try {
      target = Number(sessionStorage.getItem(`scroll:${key}`) ?? 0);
    } catch {
      return;
    }
    if (!target) return;

    let raf = 0;
    let cancelled = false;
    const deadline = performance.now() + 1000;
    const stop = () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
    const tick = () => {
      if (cancelled) return;
      window.scrollTo(0, target);
      // Short page = content still loading; keep trying until it fits or we
      // run out of patience.
      if (window.scrollY < target && performance.now() < deadline) {
        raf = requestAnimationFrame(tick);
      }
    };
    raf = requestAnimationFrame(tick);

    window.addEventListener("wheel", stop, { passive: true });
    window.addEventListener("touchstart", stop, { passive: true });
    window.addEventListener("keydown", stop);
    return () => {
      stop();
      window.removeEventListener("wheel", stop);
      window.removeEventListener("touchstart", stop);
      window.removeEventListener("keydown", stop);
    };
  }, [key, navigationType]);

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
          <ScrollRestoration />
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
            <Route path="/subscription" element={<RedirectKeepSearch to="/profile" />} />
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
            <Route path="/organization" element={<RedirectKeepSearch to="/teams" />} />
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

