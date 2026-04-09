import { useState, useEffect, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Mail, Music, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { supabase } from "@/integrations/supabase/client";

const POLL_INTERVAL = 3500;
const RESEND_COOLDOWN = 60;

const ConfirmEmail = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const email = (location.state as { email?: string })?.email || "";
  const [resendCooldown, setResendCooldown] = useState(0);
  const [resending, setResending] = useState(false);

  // Poll for email confirmation
  useEffect(() => {
    const interval = setInterval(async () => {
      const { data } = await supabase.auth.getUser();
      if (data.user?.email_confirmed_at) {
        clearInterval(interval);
        navigate("/onboarding", { replace: true });
      }
    }, POLL_INTERVAL);

    return () => clearInterval(interval);
  }, [navigate]);

  // Cooldown timer
  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setTimeout(() => setResendCooldown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [resendCooldown]);

  const handleResend = useCallback(async () => {
    if (resendCooldown > 0 || !email) return;
    setResending(true);
    await supabase.auth.resend({ type: "signup", email });
    setResending(false);
    setResendCooldown(RESEND_COOLDOWN);
  }, [email, resendCooldown]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-background via-background to-secondary/20 p-4">
      {/* Top-left branding */}
      <div className="absolute top-6 left-6 flex items-center gap-2 opacity-60">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <Music className="w-4 h-4 text-primary-foreground" />
        </div>
        <span className="text-sm font-semibold text-foreground">Msanii</span>
      </div>

      <div className="w-full max-w-md text-center space-y-6">
        {/* Mail icon */}
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mx-auto">
          <Mail className="w-8 h-8 text-primary" />
        </div>

        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-foreground">Check your inbox</h1>
          <p className="text-muted-foreground">
            We sent a confirmation link to{" "}
            {email ? <span className="font-medium text-foreground">{email}</span> : "your email"}
            . Click it to get started.
          </p>
        </div>

        {/* Resend */}
        <Button
          variant="outline"
          size="sm"
          onClick={handleResend}
          disabled={resendCooldown > 0 || resending || !email}
          className="gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${resending ? "animate-spin" : ""}`} />
          {resendCooldown > 0
            ? `Resend in ${resendCooldown}s`
            : "Resend email"}
        </Button>

        {/* Back to sign in */}
        <p className="text-sm text-muted-foreground">
          Wrong email?{" "}
          <button
            onClick={() => navigate("/auth")}
            className="text-primary hover:underline font-medium"
          >
            Back to sign in
          </button>
        </p>
      </div>
    </div>
  );
};

export default ConfirmEmail;
