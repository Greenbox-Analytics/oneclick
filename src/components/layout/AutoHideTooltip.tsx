import { ReactNode, useEffect, useRef, useState } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

/**
 * Hover tooltip for header icons that fades away on its own after 5 seconds,
 * even if the pointer stays on the icon. `disabled` suppresses it entirely —
 * pass it while an attached popover is open so the label never overlaps the
 * panel it describes.
 */
export function AutoHideTooltip({
  label,
  children,
  disabled = false,
}: {
  label: string;
  children: ReactNode;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const timer = useRef<number | null>(null);

  const clear = () => {
    if (timer.current) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  };

  const handleOpenChange = (next: boolean) => {
    clear();
    setOpen(next);
    if (next) timer.current = window.setTimeout(() => setOpen(false), 5000);
  };

  useEffect(() => clear, []);
  useEffect(() => {
    if (disabled) {
      clear();
      setOpen(false);
    }
  }, [disabled]);

  return (
    <Tooltip open={open && !disabled} onOpenChange={handleOpenChange}>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}
