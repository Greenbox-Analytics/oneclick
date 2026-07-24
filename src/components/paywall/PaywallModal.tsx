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
  managedByOrg?: boolean;
  requestUrl?: string;
}

export const PaywallModal = ({
  open,
  onClose,
  reason,
  feature,
  resource,
  managedByOrg,
  requestUrl,
}: PaywallModalProps) => (
  <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
    <DialogContent className="sm:max-w-md">
      <PaywallCard
        feature={feature}
        resource={resource}
        reason={reason}
        variant="modal"
        managedByOrg={managedByOrg}
        requestUrl={requestUrl}
      />
    </DialogContent>
  </Dialog>
);
