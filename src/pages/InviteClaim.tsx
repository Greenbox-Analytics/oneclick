import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useClaimInvitation } from "@/hooks/useRegistry";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

const InviteClaim = () => {
  const { token } = useParams<{ token: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const claimInvitation = useClaimInvitation();
  const [claimed, setClaimed] = useState(false);

  useEffect(() => {
    if (!token) return;

    if (!user) {
      navigate(`/auth?redirect=/tools/registry/invite/${token}`);
      return;
    }

    if (!claimed) {
      setClaimed(true);
      claimInvitation.mutate(token, {
        onSuccess: (data) => {
          toast.success("Invitation claimed — review your stake");
          navigate(`/tools/registry/${data.work_id}`);
        },
        onError: (error: Error) => {
          if (error.message.includes("410") || error.message.toLowerCase().includes("expired")) {
            toast.error(
              "This invitation has expired. Please ask the owner to send a new one."
            );
          } else {
            toast.error("Invalid or expired invitation");
          }
          navigate("/tools/registry");
        },
      });
    }
  }, [token, user, claimed, claimInvitation, navigate]);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
        <p className="text-muted-foreground">Claiming your invitation...</p>
      </div>
    </div>
  );
};

export default InviteClaim;
