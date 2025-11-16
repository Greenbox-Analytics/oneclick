import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Music, Calculator, User, Users, Plus } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii AI</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/")}>
            Sign Out
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">Dashboard</h2>
          <p className="text-muted-foreground">Manage your artists and calculate royalty splits</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 mb-8">
          <Card className="hover:shadow-lg transition-shadow cursor-pointer" onClick={() => navigate("/tools")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                <Calculator className="w-6 h-6 text-primary" />
              </div>
              <CardTitle>Tools</CardTitle>
              <CardDescription>Access OneClick and other management tools</CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" className="w-full">
                Open Tools
              </Button>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow cursor-pointer" onClick={() => navigate("/profile")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                <User className="w-6 h-6 text-primary" />
              </div>
              <CardTitle>Manage Profile</CardTitle>
              <CardDescription>Update your account settings and preferences</CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" className="w-full">
                View Profile
              </Button>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow cursor-pointer" onClick={() => navigate("/artists")}>
            <CardHeader>
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                <Users className="w-6 h-6 text-primary" />
              </div>
              <CardTitle>Artist Profiles</CardTitle>
              <CardDescription>View and manage your artist roster</CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" className="w-full">
                View Artists
              </Button>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Get started with common tasks</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-3">
            <Button onClick={() => navigate("/artists/new")}>
              <Plus className="w-4 h-4 mr-2" />
              Add New Artist
            </Button>
            <Button variant="outline" onClick={() => navigate("/tools")}>
              <Calculator className="w-4 h-4 mr-2" />
              OneClick
            </Button>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Dashboard;
