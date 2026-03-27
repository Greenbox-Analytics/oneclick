# User Onboarding & App Walkthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-step onboarding flow for new users (profile setup pages) and a guided walkthrough of the app's key features on first login.

**Architecture:** After signup, users are redirected to a full-screen multi-step onboarding form (inspired by Claude AI's clean signup flow) that collects their name, preferred name, industry, and company info. Once complete, they land on the Dashboard where a spotlight-style walkthrough highlights each major tool. Both onboarding completion and walkthrough completion are persisted in the `profiles` table so returning users skip them.

**Tech Stack:** React 18 + TypeScript, React Router v6, Supabase (auth + profiles table), shadcn/ui components, Tailwind CSS, Zod validation, react-hook-form

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/pages/Onboarding.tsx` | Multi-step onboarding page — manages step state, handles profile save |
| `src/components/onboarding/OnboardingProgress.tsx` | Step progress dots/bar indicator |
| `src/components/onboarding/StepWelcome.tsx` | Welcome screen with Msanii branding |
| `src/components/onboarding/StepName.tsx` | Collects first name, last name |
| `src/components/onboarding/StepPreferences.tsx` | Collects preferred name, industry, company |
| `src/components/onboarding/StepReady.tsx` | Completion screen with "Go to Dashboard" CTA |
| `src/hooks/useOnboardingStatus.ts` | Hook: queries profiles table for `onboarding_completed` flag |
| `src/components/walkthrough/WalkthroughProvider.tsx` | Context provider + overlay backdrop for walkthrough |
| `src/components/walkthrough/WalkthroughTooltip.tsx` | Positioned tooltip that highlights a target element |
| `src/hooks/useWalkthrough.ts` | Hook: walkthrough step state, next/prev/skip logic |

### Modified Files
| File | Change |
|------|--------|
| `src/integrations/supabase/types.ts` | Add `industry`, `onboarding_completed`, `walkthrough_completed` to profiles type |
| `src/App.tsx` | Add `/onboarding` route |
| `src/components/ProtectedRoute.tsx` | Check onboarding status, redirect if incomplete |
| `src/contexts/AuthContext.tsx` | Redirect to `/onboarding` after signup instead of relying on Auth page redirect |
| `src/pages/Auth.tsx` | After signup success, navigate to `/onboarding` |
| `src/pages/Dashboard.tsx` | Add `data-walkthrough` attributes to feature cards, trigger walkthrough on first visit |

### Database Changes
Add three columns to `profiles` table via Supabase SQL editor:
```sql
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS industry text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS onboarding_completed boolean DEFAULT false;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS walkthrough_completed boolean DEFAULT false;
```

---

### Task 0: Database Schema & TypeScript Types

**Goal:** Add `industry`, `onboarding_completed`, and `walkthrough_completed` columns to the profiles table and update the TypeScript types to match.

**Files:**
- Modify: `src/integrations/supabase/types.ts:83-132`
- Run SQL in Supabase dashboard (documented in steps)

**Acceptance Criteria:**
- [ ] `profiles` table has `industry` (text, nullable), `onboarding_completed` (boolean, default false), `walkthrough_completed` (boolean, default false)
- [ ] TypeScript types reflect the new columns in Row, Insert, and Update types
- [ ] Existing profiles default to `onboarding_completed = false`

**Verify:** Open Supabase Table Editor > profiles > confirm columns exist. Then `npx tsc --noEmit` passes.

**Steps:**

- [ ] **Step 1: Run SQL migration in Supabase**

Execute this SQL in the Supabase SQL Editor (Dashboard > SQL Editor > New query):

```sql
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS industry text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS onboarding_completed boolean DEFAULT false;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS walkthrough_completed boolean DEFAULT false;
```

- [ ] **Step 2: Update TypeScript types**

In `src/integrations/supabase/types.ts`, add the new fields to the `profiles` type definitions.

In the `Row` section (after `id: string`), add:
```typescript
industry: string | null
onboarding_completed: boolean
walkthrough_completed: boolean
```

In the `Insert` section (after `id: string`), add:
```typescript
industry?: string | null
onboarding_completed?: boolean
walkthrough_completed?: boolean
```

In the `Update` section (after `id?: string`), add:
```typescript
industry?: string | null
onboarding_completed?: boolean
walkthrough_completed?: boolean
```

- [ ] **Step 3: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No type errors

- [ ] **Step 4: Commit**

```bash
git add src/integrations/supabase/types.ts
git commit -m "feat: add onboarding columns to profiles schema and types"
```

---

### Task 1: Onboarding Status Hook

**Goal:** Create a hook that checks whether the current user has completed onboarding, used by ProtectedRoute to gate access.

**Files:**
- Create: `src/hooks/useOnboardingStatus.ts`
- Test: Manual — hook is consumed by ProtectedRoute in Task 5

**Acceptance Criteria:**
- [ ] Returns `{ onboardingCompleted: boolean, loading: boolean }`
- [ ] Queries `profiles` table for `onboarding_completed` column by `user.id`
- [ ] Returns `loading: true` while query is in flight
- [ ] Returns `onboardingCompleted: false` if no profile row exists

**Verify:** Import in a test component and log output — should show `{ onboardingCompleted: false, loading: false }` for a user without onboarding.

**Steps:**

- [ ] **Step 1: Create the hook file**

Create `src/hooks/useOnboardingStatus.ts`:

```typescript
import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";

interface OnboardingStatus {
  onboardingCompleted: boolean;
  walkthroughCompleted: boolean;
  loading: boolean;
}

export const useOnboardingStatus = (): OnboardingStatus => {
  const { user } = useAuth();
  const [onboardingCompleted, setOnboardingCompleted] = useState(false);
  const [walkthroughCompleted, setWalkthroughCompleted] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkStatus = async () => {
      if (!user) {
        setLoading(false);
        return;
      }

      try {
        const { data, error } = await supabase
          .from("profiles")
          .select("onboarding_completed, walkthrough_completed")
          .eq("id", user.id)
          .single();

        if (error) {
          console.error("Error checking onboarding status:", error);
          setOnboardingCompleted(false);
          setWalkthroughCompleted(false);
        } else {
          setOnboardingCompleted(data?.onboarding_completed ?? false);
          setWalkthroughCompleted(data?.walkthrough_completed ?? false);
        }
      } catch (err) {
        console.error("Unexpected error checking onboarding:", err);
      } finally {
        setLoading(false);
      }
    };

    checkStatus();
  }, [user]);

  return { onboardingCompleted, walkthroughCompleted, loading };
};
```

- [ ] **Step 2: Commit**

```bash
git add src/hooks/useOnboardingStatus.ts
git commit -m "feat: add useOnboardingStatus hook"
```

---

### Task 2: Onboarding Step Components

**Goal:** Build the four step components for the onboarding flow: Welcome, Name, Preferences, and Ready.

**Files:**
- Create: `src/components/onboarding/OnboardingProgress.tsx`
- Create: `src/components/onboarding/StepWelcome.tsx`
- Create: `src/components/onboarding/StepName.tsx`
- Create: `src/components/onboarding/StepPreferences.tsx`
- Create: `src/components/onboarding/StepReady.tsx`

**Acceptance Criteria:**
- [ ] `OnboardingProgress` renders a horizontal dot/step indicator showing current step and total steps
- [ ] `StepWelcome` shows Msanii logo, welcome message, and a "Get Started" button
- [ ] `StepName` collects first name (required) and last name (required) with validation
- [ ] `StepPreferences` collects preferred name (optional), industry (select from list), and company (optional)
- [ ] `StepReady` shows a completion message with the user's preferred name and a "Go to Dashboard" CTA
- [ ] All steps accept `onNext` callback prop; Name and Preferences accept `data`/`onUpdate` for form state
- [ ] Clean, minimal design matching Claude AI's onboarding aesthetic — centered content, generous whitespace, subtle animations

**Verify:** Each component renders without errors when imported into the Onboarding page (Task 3).

**Steps:**

- [ ] **Step 1: Create OnboardingProgress component**

Create `src/components/onboarding/OnboardingProgress.tsx`:

```typescript
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
```

- [ ] **Step 2: Create StepWelcome component**

Create `src/components/onboarding/StepWelcome.tsx`:

```typescript
import { Button } from "@/components/ui/button";
import { Music } from "lucide-react";

interface StepWelcomeProps {
  onNext: () => void;
}

const StepWelcome = ({ onNext }: StepWelcomeProps) => {
  return (
    <div className="flex flex-col items-center text-center space-y-8 animate-in fade-in duration-500">
      <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center">
        <Music className="w-10 h-10 text-primary" />
      </div>
      <div className="space-y-3">
        <h1 className="text-4xl font-bold tracking-tight text-foreground">
          Welcome to Msanii
        </h1>
        <p className="text-lg text-muted-foreground max-w-md">
          Your all-in-one platform for managing artists, royalties, contracts, and creative projects.
        </p>
      </div>
      <p className="text-sm text-muted-foreground">
        Let's get your profile set up — it only takes a minute.
      </p>
      <Button size="lg" onClick={onNext} className="px-8">
        Get Started
      </Button>
    </div>
  );
};

export default StepWelcome;
```

- [ ] **Step 3: Create StepName component**

Create `src/components/onboarding/StepName.tsx`:

```typescript
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
```

- [ ] **Step 4: Create StepPreferences component**

Create `src/components/onboarding/StepPreferences.tsx`:

```typescript
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
```

- [ ] **Step 5: Create StepReady component**

Create `src/components/onboarding/StepReady.tsx`:

```typescript
import { Button } from "@/components/ui/button";
import { CheckCircle2 } from "lucide-react";

interface StepReadyProps {
  preferredName: string;
  firstName: string;
  onFinish: () => void;
  isLoading: boolean;
}

const StepReady = ({ preferredName, firstName, onFinish, isLoading }: StepReadyProps) => {
  const displayName = preferredName.trim() || firstName.trim() || "there";

  return (
    <div className="flex flex-col items-center text-center space-y-8 animate-in fade-in duration-500">
      <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center">
        <CheckCircle2 className="w-8 h-8 text-green-500" />
      </div>
      <div className="space-y-3">
        <h2 className="text-3xl font-bold text-foreground">
          You're all set, {displayName}!
        </h2>
        <p className="text-muted-foreground max-w-md">
          Your profile is ready. We'll give you a quick tour of the tools
          when you reach the dashboard.
        </p>
      </div>
      <Button size="lg" onClick={onFinish} disabled={isLoading} className="px-8">
        {isLoading ? "Setting up..." : "Go to Dashboard"}
      </Button>
    </div>
  );
};

export default StepReady;
```

- [ ] **Step 6: Commit**

```bash
git add src/components/onboarding/
git commit -m "feat: add onboarding step components and progress indicator"
```

---

### Task 3: Onboarding Page

**Goal:** Build the main Onboarding page that orchestrates the multi-step form, manages state, and saves the profile to Supabase on completion.

**Files:**
- Create: `src/pages/Onboarding.tsx`

**Acceptance Criteria:**
- [ ] Renders a full-screen centered layout with step transitions
- [ ] Manages form state for all fields: firstName, lastName, preferredName, industry, company
- [ ] Pre-populates firstName/lastName from existing profile data (parsed from `full_name` if needed)
- [ ] On final step, upserts profile with all collected data and sets `onboarding_completed = true`
- [ ] Navigates to `/dashboard` after successful save
- [ ] Shows toast on save error

**Verify:** `npx tsc --noEmit` passes. Navigate to `/onboarding` manually — full flow renders.

**Steps:**

- [ ] **Step 1: Create the Onboarding page**

Create `src/pages/Onboarding.tsx`:

```typescript
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Music } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import OnboardingProgress from "@/components/onboarding/OnboardingProgress";
import StepWelcome from "@/components/onboarding/StepWelcome";
import StepName from "@/components/onboarding/StepName";
import StepPreferences from "@/components/onboarding/StepPreferences";
import StepReady from "@/components/onboarding/StepReady";

const TOTAL_STEPS = 4;

const Onboarding = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [currentStep, setCurrentStep] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    preferredName: "",
    industry: "",
    company: "",
  });

  // Pre-populate from existing profile data
  useEffect(() => {
    const loadExistingProfile = async () => {
      if (!user) return;

      const { data } = await supabase
        .from("profiles")
        .select("first_name, last_name, given_name, full_name, company")
        .eq("id", user.id)
        .single();

      if (data) {
        let firstName = data.first_name || "";
        let lastName = data.last_name || "";
        if (!firstName && !lastName && data.full_name) {
          const parts = data.full_name.trim().split(/\s+/);
          firstName = parts[0] || "";
          lastName = parts.slice(1).join(" ") || "";
        }

        setFormData((prev) => ({
          ...prev,
          firstName: firstName || prev.firstName,
          lastName: lastName || prev.lastName,
          preferredName: data.given_name || prev.preferredName,
          company: data.company || prev.company,
        }));
      }
    };

    loadExistingProfile();
  }, [user]);

  const handleUpdate = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleFinish = async () => {
    if (!user) return;
    setIsLoading(true);

    try {
      const fullName = `${formData.firstName} ${formData.lastName}`.trim();
      const { error } = await supabase.from("profiles").upsert({
        id: user.id,
        first_name: formData.firstName,
        last_name: formData.lastName,
        given_name: formData.preferredName || null,
        full_name: fullName,
        industry: formData.industry || null,
        company: formData.company || null,
        onboarding_completed: true,
        updated_at: new Date().toISOString(),
      });

      if (error) throw error;

      navigate("/dashboard", { replace: true });
    } catch (error: any) {
      console.error("Error saving onboarding profile:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to save profile. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-background via-background to-secondary/20 p-4">
      {/* Top-left branding */}
      <div className="absolute top-6 left-6 flex items-center gap-2 opacity-60">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <Music className="w-4 h-4 text-primary-foreground" />
        </div>
        <span className="text-sm font-semibold text-foreground">Msanii</span>
      </div>

      {/* Progress indicator — hidden on welcome step */}
      {currentStep > 0 && (
        <div className="absolute top-6">
          <OnboardingProgress currentStep={currentStep} totalSteps={TOTAL_STEPS} />
        </div>
      )}

      {/* Step content */}
      <div className="w-full max-w-lg flex items-center justify-center">
        {currentStep === 0 && (
          <StepWelcome onNext={() => setCurrentStep(1)} />
        )}
        {currentStep === 1 && (
          <StepName
            firstName={formData.firstName}
            lastName={formData.lastName}
            onUpdate={handleUpdate}
            onNext={() => setCurrentStep(2)}
            onBack={() => setCurrentStep(0)}
          />
        )}
        {currentStep === 2 && (
          <StepPreferences
            preferredName={formData.preferredName}
            industry={formData.industry}
            company={formData.company}
            onUpdate={handleUpdate}
            onNext={() => setCurrentStep(3)}
            onBack={() => setCurrentStep(1)}
          />
        )}
        {currentStep === 3 && (
          <StepReady
            preferredName={formData.preferredName}
            firstName={formData.firstName}
            onFinish={handleFinish}
            isLoading={isLoading}
          />
        )}
      </div>
    </div>
  );
};

export default Onboarding;
```

- [ ] **Step 2: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/pages/Onboarding.tsx
git commit -m "feat: add onboarding page with multi-step form flow"
```

---

### Task 4: Route Integration & Onboarding Gate

**Goal:** Wire the onboarding page into the router and modify ProtectedRoute to redirect users who haven't completed onboarding.

**Files:**
- Modify: `src/App.tsx:1-160`
- Modify: `src/components/ProtectedRoute.tsx:1-25`
- Modify: `src/pages/Auth.tsx:44-63`

**Acceptance Criteria:**
- [ ] `/onboarding` route exists and renders the Onboarding page (behind auth)
- [ ] ProtectedRoute redirects authenticated users to `/onboarding` if `onboarding_completed` is false
- [ ] ProtectedRoute does NOT redirect if user is already on `/onboarding`
- [ ] After signup, Auth page navigates to `/onboarding` instead of `/dashboard`
- [ ] Existing users with `onboarding_completed = true` go straight to their requested page
- [ ] Google OAuth redirect URL updated to `/onboarding` for new users

**Verify:** Sign up a new test user — should land on `/onboarding`. Complete onboarding — should reach `/dashboard`. Sign out, sign back in — should go to `/dashboard` directly.

**Steps:**

- [ ] **Step 1: Add onboarding route to App.tsx**

In `src/App.tsx`, add the import at the top with the other page imports:

```typescript
import Onboarding from "./pages/Onboarding";
```

Add the route after the `/auth` route (before `/dashboard`):

```tsx
<Route
  path="/onboarding"
  element={
    <ProtectedRoute skipOnboardingCheck>
      <Onboarding />
    </ProtectedRoute>
  }
/>
```

- [ ] **Step 2: Update ProtectedRoute to check onboarding status**

Replace the contents of `src/components/ProtectedRoute.tsx` with:

```typescript
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useOnboardingStatus } from "@/hooks/useOnboardingStatus";

interface ProtectedRouteProps {
  children: React.ReactNode;
  skipOnboardingCheck?: boolean;
}

export const ProtectedRoute = ({ children, skipOnboardingCheck = false }: ProtectedRouteProps) => {
  const { user, loading: authLoading } = useAuth();
  const { onboardingCompleted, loading: onboardingLoading } = useOnboardingStatus();
  const location = useLocation();

  if (authLoading || (!skipOnboardingCheck && onboardingLoading)) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  if (!skipOnboardingCheck && !onboardingCompleted && location.pathname !== "/onboarding") {
    return <Navigate to="/onboarding" replace />;
  }

  return <>{children}</>;
};
```

- [ ] **Step 3: Update Auth.tsx signup redirect**

In `src/pages/Auth.tsx`, change the `handleSignUp` function. After the `await signUp(...)` call succeeds, navigate to `/onboarding` instead of showing just a toast:

Replace the success block in `handleSignUp` (lines ~49-53):

```typescript
await signUp(signUpEmail, signUpPassword, signUpName);
toast({
  title: "Success!",
  description: "Account created! Please check your email to verify your account.",
});
```

With:

```typescript
await signUp(signUpEmail, signUpPassword, signUpName);
toast({
  title: "Welcome!",
  description: "Account created! Let's set up your profile.",
});
navigate("/onboarding");
```

- [ ] **Step 4: Update Google OAuth redirect for new users**

In `src/contexts/AuthContext.tsx`, update the `signInWithGoogle` redirect to go to `/onboarding` so that new Google sign-up users hit the onboarding flow. Change line 75:

```typescript
redirectTo: `${window.location.origin}/dashboard`,
```

To:

```typescript
redirectTo: `${window.location.origin}/onboarding`,
```

Note: The ProtectedRoute will handle the logic — if a returning Google user already has `onboarding_completed = true`, they'll be allowed through to the dashboard from the onboarding page check. The Onboarding page itself should also check if already completed and redirect:

Add this to the top of the `Onboarding` component in `src/pages/Onboarding.tsx`, after the `useEffect` for loading profile data:

```typescript
// Redirect if onboarding already completed
useEffect(() => {
  const checkIfCompleted = async () => {
    if (!user) return;
    const { data } = await supabase
      .from("profiles")
      .select("onboarding_completed")
      .eq("id", user.id)
      .single();

    if (data?.onboarding_completed) {
      navigate("/dashboard", { replace: true });
    }
  };
  checkIfCompleted();
}, [user, navigate]);
```

- [ ] **Step 5: Verify the full flow**

Run: `npm run dev`
1. Open app in incognito
2. Go to `/auth`, create a new account
3. Should redirect to `/onboarding`
4. Complete all steps
5. Should reach `/dashboard`
6. Refresh — should stay on `/dashboard` (not redirect to onboarding)

- [ ] **Step 6: Commit**

```bash
git add src/App.tsx src/components/ProtectedRoute.tsx src/pages/Auth.tsx src/contexts/AuthContext.tsx src/pages/Onboarding.tsx
git commit -m "feat: integrate onboarding route and gate protected routes"
```

---

### Task 5: Walkthrough Hook & Provider

**Goal:** Build a walkthrough system that highlights dashboard elements with a spotlight overlay and explanatory tooltips.

**Files:**
- Create: `src/hooks/useWalkthrough.ts`
- Create: `src/components/walkthrough/WalkthroughProvider.tsx`
- Create: `src/components/walkthrough/WalkthroughTooltip.tsx`

**Acceptance Criteria:**
- [ ] `useWalkthrough` manages current step index, provides next/skip/complete actions
- [ ] Walkthrough steps are defined as a static config array with `targetSelector`, `title`, `description`, and `placement`
- [ ] `WalkthroughProvider` renders a backdrop overlay with a spotlight cutout around the target element
- [ ] `WalkthroughTooltip` positions itself relative to the highlighted element
- [ ] Completing or skipping sets `walkthrough_completed = true` in Supabase
- [ ] Walkthrough auto-starts when Dashboard mounts and `walkthrough_completed` is false

**Verify:** `npx tsc --noEmit` passes. Walkthrough triggers on Dashboard for users with `walkthrough_completed = false`.

**Steps:**

- [ ] **Step 1: Create walkthrough step configuration and hook**

Create `src/hooks/useWalkthrough.ts`:

```typescript
import { useState, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";

export interface WalkthroughStep {
  targetSelector: string;
  title: string;
  description: string;
  placement: "top" | "bottom" | "left" | "right";
}

export const WALKTHROUGH_STEPS: WalkthroughStep[] = [
  {
    targetSelector: '[data-walkthrough="tools"]',
    title: "Tools",
    description:
      "Access OneClick for rapid document processing, Zoe AI for contract analysis, and Split Sheets for royalty management.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="artists"]',
    title: "Artist Profiles",
    description:
      "Create and manage your artist roster — store DSP links, social media, contracts, and project files all in one place.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="workspace"]',
    title: "Workspace",
    description:
      "Organize projects with Kanban boards, connect Google Drive, Slack, Notion, and Monday.com integrations.",
    placement: "bottom",
  },
  {
    targetSelector: '[data-walkthrough="portfolio"]',
    title: "Portfolio",
    description:
      "See everything at a glance — your artists, projects, and documents organized by year and category.",
    placement: "bottom",
  },
];

interface UseWalkthroughReturn {
  isActive: boolean;
  currentStepIndex: number;
  currentStep: WalkthroughStep | null;
  totalSteps: number;
  start: () => void;
  next: () => void;
  skip: () => void;
}

export const useWalkthrough = (): UseWalkthroughReturn => {
  const { user } = useAuth();
  const [isActive, setIsActive] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  const markCompleted = useCallback(async () => {
    if (!user) return;
    await supabase
      .from("profiles")
      .update({ walkthrough_completed: true, updated_at: new Date().toISOString() })
      .eq("id", user.id);
  }, [user]);

  const start = useCallback(() => {
    setCurrentStepIndex(0);
    setIsActive(true);
  }, []);

  const next = useCallback(() => {
    if (currentStepIndex < WALKTHROUGH_STEPS.length - 1) {
      setCurrentStepIndex((prev) => prev + 1);
    } else {
      setIsActive(false);
      markCompleted();
    }
  }, [currentStepIndex, markCompleted]);

  const skip = useCallback(() => {
    setIsActive(false);
    markCompleted();
  }, [markCompleted]);

  return {
    isActive,
    currentStepIndex,
    currentStep: isActive ? WALKTHROUGH_STEPS[currentStepIndex] : null,
    totalSteps: WALKTHROUGH_STEPS.length,
    start,
    next,
    skip,
  };
};
```

- [ ] **Step 2: Create WalkthroughTooltip component**

Create `src/components/walkthrough/WalkthroughTooltip.tsx`:

```typescript
import { useEffect, useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import type { WalkthroughStep } from "@/hooks/useWalkthrough";

interface WalkthroughTooltipProps {
  step: WalkthroughStep;
  stepIndex: number;
  totalSteps: number;
  onNext: () => void;
  onSkip: () => void;
}

const WalkthroughTooltip = ({
  step,
  stepIndex,
  totalSteps,
  onNext,
  onSkip,
}: WalkthroughTooltipProps) => {
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const tooltipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const target = document.querySelector(step.targetSelector);
    if (!target) return;

    const rect = target.getBoundingClientRect();
    const tooltipWidth = 320;
    const gap = 12;

    let top = 0;
    let left = 0;

    switch (step.placement) {
      case "bottom":
        top = rect.bottom + gap + window.scrollY;
        left = rect.left + rect.width / 2 - tooltipWidth / 2 + window.scrollX;
        break;
      case "top":
        top = rect.top - gap + window.scrollY;
        left = rect.left + rect.width / 2 - tooltipWidth / 2 + window.scrollX;
        break;
      case "right":
        top = rect.top + rect.height / 2 + window.scrollY;
        left = rect.right + gap + window.scrollX;
        break;
      case "left":
        top = rect.top + rect.height / 2 + window.scrollY;
        left = rect.left - tooltipWidth - gap + window.scrollX;
        break;
    }

    // Clamp to viewport
    left = Math.max(16, Math.min(left, window.innerWidth - tooltipWidth - 16));

    setPosition({ top, left });

    // Scroll target into view
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [step]);

  const isLast = stepIndex === totalSteps - 1;

  return (
    <div
      ref={tooltipRef}
      className="fixed z-[60] w-80 bg-card border border-border rounded-xl shadow-2xl p-5 animate-in fade-in slide-in-from-bottom-2 duration-300"
      style={{ top: position.top, left: position.left }}
    >
      <div className="space-y-2 mb-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-foreground">{step.title}</h3>
          <span className="text-xs text-muted-foreground">
            {stepIndex + 1} / {totalSteps}
          </span>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          {step.description}
        </p>
      </div>
      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onSkip}>
          Skip tour
        </Button>
        <Button size="sm" onClick={onNext}>
          {isLast ? "Finish" : "Next"}
        </Button>
      </div>
    </div>
  );
};

export default WalkthroughTooltip;
```

- [ ] **Step 3: Create WalkthroughProvider component**

Create `src/components/walkthrough/WalkthroughProvider.tsx`:

```typescript
import { useEffect, useState } from "react";
import WalkthroughTooltip from "./WalkthroughTooltip";
import type { WalkthroughStep } from "@/hooks/useWalkthrough";

interface WalkthroughProviderProps {
  isActive: boolean;
  currentStep: WalkthroughStep | null;
  currentStepIndex: number;
  totalSteps: number;
  onNext: () => void;
  onSkip: () => void;
}

const WalkthroughProvider = ({
  isActive,
  currentStep,
  currentStepIndex,
  totalSteps,
  onNext,
  onSkip,
}: WalkthroughProviderProps) => {
  const [spotlightRect, setSpotlightRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    if (!isActive || !currentStep) {
      setSpotlightRect(null);
      return;
    }

    const updateSpotlight = () => {
      const target = document.querySelector(currentStep.targetSelector);
      if (target) {
        setSpotlightRect(target.getBoundingClientRect());
      }
    };

    updateSpotlight();

    window.addEventListener("resize", updateSpotlight);
    window.addEventListener("scroll", updateSpotlight);

    return () => {
      window.removeEventListener("resize", updateSpotlight);
      window.removeEventListener("scroll", updateSpotlight);
    };
  }, [isActive, currentStep]);

  if (!isActive || !currentStep) return null;

  const padding = 8;

  return (
    <>
      {/* Backdrop with spotlight cutout */}
      <div className="fixed inset-0 z-50 pointer-events-auto">
        <svg className="absolute inset-0 w-full h-full">
          <defs>
            <mask id="walkthrough-mask">
              <rect width="100%" height="100%" fill="white" />
              {spotlightRect && (
                <rect
                  x={spotlightRect.left - padding}
                  y={spotlightRect.top - padding}
                  width={spotlightRect.width + padding * 2}
                  height={spotlightRect.height + padding * 2}
                  rx="12"
                  fill="black"
                />
              )}
            </mask>
          </defs>
          <rect
            width="100%"
            height="100%"
            fill="rgba(0,0,0,0.5)"
            mask="url(#walkthrough-mask)"
          />
        </svg>
      </div>

      {/* Tooltip */}
      <WalkthroughTooltip
        step={currentStep}
        stepIndex={currentStepIndex}
        totalSteps={totalSteps}
        onNext={onNext}
        onSkip={onSkip}
      />
    </>
  );
};

export default WalkthroughProvider;
```

- [ ] **Step 4: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/hooks/useWalkthrough.ts src/components/walkthrough/
git commit -m "feat: add walkthrough system with spotlight overlay and tooltip"
```

---

### Task 6: Dashboard Walkthrough Integration

**Goal:** Add walkthrough data attributes to Dashboard feature cards and trigger the walkthrough for first-time users.

**Files:**
- Modify: `src/pages/Dashboard.tsx:200-260`

**Acceptance Criteria:**
- [ ] Each of the 4 main feature cards has a `data-walkthrough` attribute matching the walkthrough step config
- [ ] Walkthrough auto-starts when `walkthroughCompleted` is false (checked via `useOnboardingStatus`)
- [ ] Walkthrough renders via `WalkthroughProvider`
- [ ] After walkthrough completes, overlay disappears and dashboard is fully interactive

**Verify:** `npm run dev` — new user who completed onboarding sees walkthrough on Dashboard. After completing/skipping tour, refreshing does not re-trigger it.

**Steps:**

- [ ] **Step 1: Add imports to Dashboard.tsx**

Add these imports at the top of `src/pages/Dashboard.tsx`:

```typescript
import { useOnboardingStatus } from "@/hooks/useOnboardingStatus";
import { useWalkthrough } from "@/hooks/useWalkthrough";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
```

- [ ] **Step 2: Initialize walkthrough in Dashboard component**

Inside the `Dashboard` component, after the existing `useEffect` hooks, add:

```typescript
const { walkthroughCompleted } = useOnboardingStatus();
const walkthrough = useWalkthrough();

// Auto-start walkthrough for first-time users
useEffect(() => {
  if (walkthroughCompleted === false && !walkthrough.isActive) {
    // Small delay to ensure DOM elements are rendered
    const timer = setTimeout(() => walkthrough.start(), 500);
    return () => clearTimeout(timer);
  }
}, [walkthroughCompleted]);
```

- [ ] **Step 3: Add data-walkthrough attributes to feature cards**

In the grid section of Dashboard.tsx (the 4 main cards around line 200-260), add `data-walkthrough` attributes:

On the Tools card (the first `<Card>`), add `data-walkthrough="tools"`:
```tsx
<Card data-walkthrough="tools" className="flex flex-col border-primary/40 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => handleNavigate("/tools", "Tools")}>
```

On the Artist Profiles card, add `data-walkthrough="artists"`:
```tsx
<Card data-walkthrough="artists" className="flex flex-col border-primary/40 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => handleNavigate("/artists", "Artist Profiles")}>
```

On the Workspace card, add `data-walkthrough="workspace"`:
```tsx
<Card data-walkthrough="workspace" className="flex flex-col border-primary/40 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => handleNavigate("/workspace", "Workspace")}>
```

On the Portfolio card, add `data-walkthrough="portfolio"`:
```tsx
<Card data-walkthrough="portfolio" className="flex flex-col border-primary/40 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => handleNavigate("/portfolio", "Portfolio")}>
```

- [ ] **Step 4: Render WalkthroughProvider in Dashboard JSX**

Just before the closing `</div>` of the Dashboard return, add:

```tsx
<WalkthroughProvider
  isActive={walkthrough.isActive}
  currentStep={walkthrough.currentStep}
  currentStepIndex={walkthrough.currentStepIndex}
  totalSteps={walkthrough.totalSteps}
  onNext={walkthrough.next}
  onSkip={walkthrough.skip}
/>
```

- [ ] **Step 5: Verify full flow end-to-end**

Run: `npm run dev`

1. Create a new account
2. Complete onboarding steps
3. Land on Dashboard — walkthrough should auto-start
4. Step through all 4 spotlight highlights
5. After finishing, refresh — walkthrough should NOT re-trigger
6. Sign out, sign back in — should go to Dashboard (no onboarding, no walkthrough)

- [ ] **Step 6: Commit**

```bash
git add src/pages/Dashboard.tsx
git commit -m "feat: integrate walkthrough tour into Dashboard for first-time users"
```

---

## Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 0 | Database schema & TypeScript types | — |
| 1 | Onboarding status hook | 0 |
| 2 | Onboarding step components | — |
| 3 | Onboarding page | 1, 2 |
| 4 | Route integration & onboarding gate | 1, 3 |
| 5 | Walkthrough hook & provider | 0 |
| 6 | Dashboard walkthrough integration | 4, 5 |
