import { Dialog, DialogContent } from "@/components/ui/dialog";
import { PaywallCard } from "./PaywallCard";
import type { GatedFeature, CountableResource } from "@/hooks/useEntitlements";

interface PaywallModalProps {
  open: boolean;
  onClose: () => void;
  reason?: string;
  feature?: GatedFeature;
  resource?: CountableResource;
  /** Licensing Phase B (plan Task 13) — see PaywallCard. */
  creditWall?: boolean;
  managedByOrg?: boolean;
  capReached?: boolean;
  requestUrl?: string;
}

export const PaywallModal = ({
  open,
  onClose,
  reason,
  feature,
  resource,
  creditWall,
  managedByOrg,
  capReached,
  requestUrl,
}: PaywallModalProps) => (
  <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
    <DialogContent className="sm:max-w-md">
      <PaywallCard
        feature={feature}
        resource={resource}
        reason={reason}
        variant="modal"
        creditWall={creditWall}
        managedByOrg={managedByOrg}
        capReached={capReached}
        requestUrl={requestUrl}
      />
    </DialogContent>
  </Dialog>
);
