import { lazy, Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { ProtectedRoute } from "@/components/ProtectedRoute";

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
const WorkspaceBoards = lazy(() => import("./pages/WorkspaceBoards"));
const Portfolio = lazy(() => import("./pages/Portfolio"));
const SplitSheet = lazy(() => import("./pages/SplitSheet"));
const Onboarding = lazy(() => import("./pages/Onboarding"));
const Documentation = lazy(() => import("./pages/Documentation"));
const Registry = lazy(() => import("./pages/Registry"));
const WorkDetail = lazy(() => import("./pages/WorkDetail"));
const InviteClaim = lazy(() => import("./pages/InviteClaim"));
const ProjectDetail = lazy(() => import("./pages/ProjectDetail"));

const queryClient = new QueryClient();

const PageLoader = () => (
  <div className="min-h-screen bg-background flex items-center justify-center">
    <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
  </div>
);

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
          <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/auth" element={<Auth />} />
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
            <Route
              path="/workspace"
              element={
                <ProtectedRoute>
                  <Workspace />
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
            <Route path="/tools/registry" element={<ProtectedRoute><Registry /></ProtectedRoute>} />
            <Route path="/tools/registry/invite/:token" element={<ProtectedRoute><InviteClaim /></ProtectedRoute>} />
            <Route path="/tools/registry/:workId" element={<ProtectedRoute><WorkDetail /></ProtectedRoute>} />
            <Route path="/projects/:projectId" element={<ProtectedRoute><ProjectDetail /></ProtectedRoute>} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
          </Suspense>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;

