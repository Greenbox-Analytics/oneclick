import { Dialog, DialogContent } from "@/components/ui/dialog";
import { PaywallCard } from "./PaywallCard";
import type { GatedFeature, CountableResource } from "@/hooks/useEntitlements";

interface PaywallModalProps {
  open: boolean;
  onClose: () => void;
  reason?: string;
  feature?: GatedFeature;
  resource?: CountableResource;
}

export const PaywallModal = ({
  open,
  onClose,
  reason,
  feature,
  resource,
}: PaywallModalProps) => (
  <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
    <DialogContent className="sm:max-w-md">
      <PaywallCard
        feature={feature}
        resource={resource}
        reason={reason}
        variant="modal"
      />
    </DialogContent>
  </Dialog>
);
