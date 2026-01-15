import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Music, Calculator, ArrowRight, ArrowLeft, Bot } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Tools = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div 
            className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" 
            onClick={() => navigate("/dashboard")}
          >
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/dashboard")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">Tools</h2>
          <p className="text-muted-foreground">Select a tool to manage your music data</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* OneClick Tool Card */}
          <Card className="hover:border-primary/50 transition-colors cursor-pointer group" onClick={() => navigate("/tools/oneclick")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                <Calculator className="w-6 h-6 text-primary" />
              </div>
              <CardTitle className="flex items-center gap-2">
                OneClick
                <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity transform translate-x-[-10px] group-hover:translate-x-0" />
              </CardTitle>
              <CardDescription>
                Calculate royalty splits and manage contracts for your artists in one click.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="ghost" className="w-full justify-start p-0 hover:bg-transparent hover:text-primary">
                Launch Tool →
              </Button>
            </CardContent>
          </Card>

          {/* Zoe AI Chatbot Tool Card */}
          <Card className="hover:border-primary/50 transition-colors cursor-pointer group" onClick={() => navigate("/tools/zoe")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                <Bot className="w-6 h-6 text-primary" />
              </div>
              <CardTitle className="flex items-center gap-2">
                Zoe
                <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity transform translate-x-[-10px] group-hover:translate-x-0" />
              </CardTitle>
              <CardDescription>
                Ask questions about your contracts and get AI-powered answers with source citations.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="ghost" className="w-full justify-start p-0 hover:bg-transparent hover:text-primary">
                Launch Tool →
              </Button>
            </CardContent>
          </Card>

          {/* Future tools placeholder */}
          <Card className="border-dashed opacity-60">
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-muted flex items-center justify-center mb-4">
                <Music className="w-6 h-6 text-muted-foreground" />
              </div>
              <CardTitle>Coming Soon</CardTitle>
              <CardDescription>
                More powerful tools for music management are currently in development.
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Tools;
