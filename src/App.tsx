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
import DocumentUpload from "./pages/DocumentUpload";
import Profile from "./pages/Profile";
import NotFound from "./pages/NotFound";

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
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
