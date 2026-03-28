import { HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ToolHelpButtonProps {
  onClick: () => void;
}

const ToolHelpButton = ({ onClick }: ToolHelpButtonProps) => {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClick}
            className="h-9 w-9"
          >
            <HelpCircle className="w-5 h-5 text-muted-foreground" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Replay tour</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default ToolHelpButton;
