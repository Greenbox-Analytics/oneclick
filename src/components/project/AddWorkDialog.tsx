import { AddWorkWizard } from "@/components/registry/AddWorkWizard";

interface AddWorkDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Empty string = registry-launched (wizard shows destination step). */
  projectId: string;
  artistId: string;
}

/**
 * Thin wrapper around the registry's Add Work wizard. Keeping this file at the
 * project-component path so existing call sites (WorksTab, RegistryDashboard)
 * don't need to change.
 */
export default function AddWorkDialog({
  open,
  onOpenChange,
  projectId,
  artistId,
}: AddWorkDialogProps) {
  return (
    <AddWorkWizard
      open={open}
      onClose={() => onOpenChange(false)}
      initialProjectId={projectId || undefined}
      initialArtistId={artistId || undefined}
    />
  );
}
