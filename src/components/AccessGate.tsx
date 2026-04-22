import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Music, Lock } from "lucide-react";

const STORAGE_KEY = "msanii_access_granted";

export const isAccessGranted = () =>
  localStorage.getItem(STORAGE_KEY) === "true";

export const AccessGate = () => {
  const navigate = useNavigate();
  const [code, setCode] = useState("");
  const [error, setError] = useState("");

  // If already granted, redirect to landing page
  useEffect(() => {
    if (isAccessGranted()) {
      navigate("/home", { replace: true });
    }
  }, [navigate]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (code === import.meta.env.VITE_ACCESS_CODE) {
      localStorage.setItem(STORAGE_KEY, "true");
      navigate("/home", { replace: true });
    } else {
      setError("Invalid access code. Please try again.");
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <Card className="w-full max-w-md p-8 text-center">
        <div className="flex items-center justify-center gap-2 mb-6">
          <Music className="w-8 h-8 text-primary" />
          <span className="text-2xl font-bold text-foreground">Msanii</span>
        </div>
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary/10 mb-4">
          <Lock className="w-6 h-6 text-primary" />
        </div>
        <h2 className="text-xl font-semibold text-foreground mb-2">
          Early Access
        </h2>
        <p className="text-sm text-muted-foreground mb-6">
          Enter the access code to continue.
        </p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            type="password"
            placeholder="Access code"
            value={code}
            onChange={(e) => {
              setCode(e.target.value);
              setError("");
            }}
            className="text-center"
            autoFocus
          />
          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
          <Button type="submit" className="w-full">
            Continue
          </Button>
        </form>
      </Card>
    </div>
  );
};
