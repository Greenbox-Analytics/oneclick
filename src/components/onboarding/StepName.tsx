import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface StepNameProps {
  firstName: string;
  lastName: string;
  onUpdate: (field: string, value: string) => void;
  onNext: () => void;
  onBack: () => void;
}

const StepName = ({ firstName, lastName, onUpdate, onNext, onBack }: StepNameProps) => {
  const isValid = firstName.trim().length > 0 && lastName.trim().length > 0;

  return (
    <div className="flex flex-col items-center space-y-8 w-full max-w-sm animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold text-foreground">What's your name?</h2>
        <p className="text-muted-foreground">We'll use this across your Msanii account.</p>
      </div>

      <div className="w-full space-y-4">
        <div className="space-y-2">
          <Label htmlFor="onboard-first-name">First name</Label>
          <Input
            id="onboard-first-name"
            value={firstName}
            onChange={(e) => onUpdate("firstName", e.target.value)}
            placeholder="e.g. Amara"
            autoFocus
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="onboard-last-name">Last name</Label>
          <Input
            id="onboard-last-name"
            value={lastName}
            onChange={(e) => onUpdate("lastName", e.target.value)}
            placeholder="e.g. Osei"
          />
        </div>
      </div>

      <div className="flex gap-3 w-full">
        <Button variant="ghost" onClick={onBack} className="flex-1">
          Back
        </Button>
        <Button onClick={onNext} disabled={!isValid} className="flex-1">
          Continue
        </Button>
      </div>
    </div>
  );
};

export default StepName;
