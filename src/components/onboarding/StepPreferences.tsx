import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const INDUSTRIES = [
  "Music Production",
  "Artist Management",
  "Record Label",
  "Music Publishing",
  "Live Events & Touring",
  "Music Distribution",
  "Music Licensing",
  "Audio Engineering",
  "Entertainment Law",
  "Independent Artist",
  "Other",
] as const;

interface StepPreferencesProps {
  preferredName: string;
  industry: string;
  company: string;
  onUpdate: (field: string, value: string) => void;
  onNext: () => void;
  onBack: () => void;
}

const StepPreferences = ({
  preferredName,
  industry,
  company,
  onUpdate,
  onNext,
  onBack,
}: StepPreferencesProps) => {
  return (
    <div className="flex flex-col items-center space-y-8 w-full max-w-sm animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-bold text-foreground">A few more details</h2>
        <p className="text-muted-foreground">Help us personalize your experience.</p>
      </div>

      <div className="w-full space-y-4">
        <div className="space-y-2">
          <Label htmlFor="onboard-preferred-name">
            What should we call you?
          </Label>
          <Input
            id="onboard-preferred-name"
            value={preferredName}
            onChange={(e) => onUpdate("preferredName", e.target.value)}
            placeholder="Nickname or preferred name (optional)"
          />
          <p className="text-xs text-muted-foreground">
            We'll use this to greet you throughout the app.
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="onboard-industry">What industry are you in?</Label>
          <Select
            value={industry}
            onValueChange={(value) => onUpdate("industry", value)}
          >
            <SelectTrigger id="onboard-industry">
              <SelectValue placeholder="Select your industry" />
            </SelectTrigger>
            <SelectContent>
              {INDUSTRIES.map((ind) => (
                <SelectItem key={ind} value={ind}>
                  {ind}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="onboard-company">Company or organization</Label>
          <Input
            id="onboard-company"
            value={company}
            onChange={(e) => onUpdate("company", e.target.value)}
            placeholder="e.g. Greenbox Analytics (optional)"
          />
        </div>
      </div>

      <div className="flex gap-3 w-full">
        <Button variant="ghost" onClick={onBack} className="flex-1">
          Back
        </Button>
        <Button onClick={onNext} className="flex-1">
          Continue
        </Button>
      </div>
    </div>
  );
};

export default StepPreferences;
