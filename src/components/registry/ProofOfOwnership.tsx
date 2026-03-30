import { useExportProof } from "@/hooks/useRegistry";
import { Button } from "@/components/ui/button";
import { Download, Loader2 } from "lucide-react";

interface Props {
  workId: string;
}

export default function ProofOfOwnership({ workId }: Props) {
  const exportProof = useExportProof();
  return (
    <Button
      variant="default" size="sm"
      onClick={() => exportProof.mutate(workId)}
      disabled={exportProof.isPending}
    >
      {exportProof.isPending ? (
        <Loader2 className="w-4 h-4 mr-1 animate-spin" />
      ) : (
        <Download className="w-4 h-4 mr-1" />
      )}
      Export Proof
    </Button>
  );
}
