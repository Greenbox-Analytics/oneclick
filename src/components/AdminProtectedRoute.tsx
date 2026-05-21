import { Navigate } from "react-router-dom";
import { useIsAdmin } from "@/hooks/useAdmin";

interface Props {
  children: React.ReactNode;
}

export const AdminProtectedRoute = ({ children }: Props) => {
  const { isAdmin, loading } = useIsAdmin();

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!isAdmin) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};
