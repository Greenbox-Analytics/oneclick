interface OnboardingProgressProps {
  currentStep: number;
  totalSteps: number;
}

const OnboardingProgress = ({ currentStep, totalSteps }: OnboardingProgressProps) => {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: totalSteps }, (_, i) => (
        <div
          key={i}
          className={`h-2 rounded-full transition-all duration-300 ${
            i === currentStep
              ? "w-8 bg-primary"
              : i < currentStep
              ? "w-2 bg-primary/60"
              : "w-2 bg-muted-foreground/20"
          }`}
        />
      ))}
    </div>
  );
};

export default OnboardingProgress;
