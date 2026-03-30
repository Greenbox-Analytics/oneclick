import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import Index from "./pages/Index";
import Auth from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import Artists from "./pages/Artists";
import ArtistProfile from "./pages/ArtistProfile";
import NewArtist from "./pages/NewArtist";
import Tools from "./pages/Tools";
import OneClick from "./pages/OneClick";
import OneClickDocuments from "./pages/OneClickDocuments";
import Zoe from "./pages/Zoe";
import Profile from "./pages/Profile";
import Workspace from "./pages/Workspace";
import WorkspaceBoards from "./pages/WorkspaceBoards";
import Portfolio from "./pages/Portfolio";
import SplitSheet from "./pages/SplitSheet";
import Onboarding from "./pages/Onboarding";
import NotFound from "./pages/NotFound";
import Documentation from "./pages/Documentation";
import Registry from "./pages/Registry";
import WorkDetail from "./pages/WorkDetail";
import InviteClaim from "./pages/InviteClaim";
import ProjectDetail from "./pages/ProjectDetail";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
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
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;

