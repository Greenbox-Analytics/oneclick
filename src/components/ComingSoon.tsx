import { useNavigate } from "react-router-dom";
import { BookOpen } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { PageHeader } from "@/components/layout/PageHeader";
import { useSmartBack } from "@/hooks/useSmartBack";

interface ComingSoonProps {
  icon: LucideIcon;
  title: string;
  message?: string;
}

export function ComingSoon({ icon: Icon, title, message }: ComingSoonProps) {
  const navigate = useNavigate();
  const goBack = useSmartBack("/tools");

  return (
    <div className="min-h-screen bg-background">
      <PageHeader
        actions={
          <>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <Button variant="outline" className="hidden md:inline-flex" onClick={goBack}>
              Back
            </Button>
          </>
        }
      />

      <main className="container mx-auto px-4 py-16 max-w-2xl">
        <Card className="border border-border">
          <CardContent className="flex flex-col items-center text-center py-16 px-6">
            <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-5">
              <Icon className="w-8 h-8 text-primary" />
            </div>
            <h2 className="text-2xl font-bold text-foreground mb-2">{title}</h2>
            <p className="text-sm text-muted-foreground max-w-md mb-6">
              {message || "This feature is coming soon. We're putting the finishing touches on it."}
            </p>
            <Button onClick={() => navigate("/tools")}>Back to Tools</Button>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
