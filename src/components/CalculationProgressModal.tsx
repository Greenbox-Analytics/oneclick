import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { CheckCircle2, FileText, Calculator, Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface CalculationProgressModalProps {
  isOpen: boolean;
  progress: number;
  stage: string;
  message: string;
}

const CalculationProgressModal = ({ isOpen, progress, stage, message }: CalculationProgressModalProps) => {
  // Determine which stages are complete, in progress, or pending
  const stages = [
    { id: 'downloading', label: 'Downloading files', icon: FileText },
    { id: 'extracting', label: 'Extracting data', icon: FileText },
    { id: 'calculating', label: 'Calculating payments', icon: Calculator },
  ];

  const getStageStatus = (stageId: string) => {
    if (stage.includes('download')) {
      if (stageId === 'downloading') return 'active';
      return 'pending';
    } else if (stage.includes('extract') || stage.includes('parties') || stage.includes('works') || stage.includes('royalty') || stage.includes('summary')) {
      if (stageId === 'downloading') return 'complete';
      if (stageId === 'extracting') return 'active';
      return 'pending';
    } else if (stage.includes('processing') || stage.includes('calculating')) {
      if (stageId === 'downloading' || stageId === 'extracting') return 'complete';
      if (stageId === 'calculating') return 'active';
      return 'pending';
    }
    return 'pending';
  };

  return (
    <Dialog open={isOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-center">Calculating Royalties</DialogTitle>
          <DialogDescription className="text-center">
            Please wait while we process your documents
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6 py-4">
          {/* Circular Progress */}
          <div className="flex flex-col items-center justify-center">
            <div className="relative w-32 h-32">
              {/* Background circle */}
              <svg className="w-32 h-32 transform -rotate-90">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="none"
                  className="text-muted"
                />
                {/* Progress circle */}
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 56}`}
                  strokeDashoffset={`${2 * Math.PI * 56 * (1 - progress / 100)}`}
                  className="text-primary transition-all duration-500 ease-out"
                  strokeLinecap="round"
                />
              </svg>
              {/* Percentage text */}
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-foreground">{Math.round(progress)}%</span>
              </div>
            </div>
            
            {/* Current stage message */}
            <p className="mt-4 text-sm font-medium text-center text-muted-foreground">
              {message}
            </p>
          </div>

          {/* Stage indicators */}
          <div className="space-y-3">
            {stages.map((stageItem) => {
              const status = getStageStatus(stageItem.id);
              const Icon = stageItem.icon;
              
              return (
                <div key={stageItem.id} className="flex items-center gap-3">
                  <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                    status === 'complete' 
                      ? 'bg-primary text-primary-foreground' 
                      : status === 'active'
                      ? 'bg-primary/20 text-primary'
                      : 'bg-muted text-muted-foreground'
                  }`}>
                    {status === 'complete' ? (
                      <CheckCircle2 className="w-4 h-4" />
                    ) : status === 'active' ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Icon className="w-4 h-4" />
                    )}
                  </div>
                  <span className={`text-sm ${
                    status === 'complete' || status === 'active'
                      ? 'text-foreground font-medium'
                      : 'text-muted-foreground'
                  }`}>
                    {stageItem.label}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Progress bar as backup visual */}
          <Progress value={progress} className="w-full" />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default CalculationProgressModal;
