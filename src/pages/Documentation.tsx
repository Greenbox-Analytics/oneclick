import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  Music,
  ArrowLeft,
  BookOpen,
  Calculator,
  Bot,
  FileText,
  Users,
  LayoutGrid,
  Folder,
  Phone,
  Lightbulb,
  Rocket,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

interface Section {
  id: string;
  label: string;
  icon: React.ElementType;
}

const SECTIONS: Section[] = [
  { id: "getting-started", label: "Getting Started", icon: Rocket },
  { id: "oneclick", label: "OneClick", icon: Calculator },
  { id: "zoe", label: "Zoe AI", icon: Bot },
  { id: "split-sheet", label: "Split Sheet", icon: FileText },
  { id: "artist-management", label: "Artist Management", icon: Users },
  { id: "workspace", label: "Workspace", icon: LayoutGrid },
  { id: "portfolio", label: "Portfolio", icon: Folder },
  { id: "contacts-payments", label: "Contacts & Payments", icon: Phone },
  { id: "best-practices", label: "Best Practices", icon: Lightbulb },
];

const Documentation = () => {
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState("getting-started");

  useEffect(() => {
    const handleScroll = () => {
      const sectionElements = SECTIONS.map((s) => ({
        id: s.id,
        el: document.getElementById(s.id),
      }));
      for (const { id, el } of sectionElements) {
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= 120 && rect.bottom > 120) {
            setActiveSection(id);
            break;
          }
        }
      }
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveSection(id);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/")}
          >
            <Music className="w-8 h-8" />
            <span className="text-xl font-bold text-foreground">Msanii</span>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={() => navigate("/dashboard")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Dashboard
            </Button>
            <Button onClick={() => navigate("/auth")}>Sign In</Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <BookOpen className="w-8 h-8 text-primary" />
            <h1 className="text-3xl font-bold text-foreground">Documentation</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Learn how to use Msanii to manage your music business effectively.
          </p>
        </div>

        <div className="flex gap-8">
          {/* Sidebar Navigation - hidden on mobile */}
          <nav className="hidden lg:block w-64 shrink-0">
            <div className="sticky top-24 space-y-1">
              {SECTIONS.map((section) => {
                const Icon = section.icon;
                return (
                  <button
                    key={section.id}
                    onClick={() => scrollToSection(section.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left ${
                      activeSection === section.id
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    {section.label}
                  </button>
                );
              })}
            </div>
          </nav>

          {/* Mobile Navigation - visible on small screens */}
          <div className="lg:hidden w-full mb-6 overflow-x-auto">
            <div className="flex gap-2 pb-2">
              {SECTIONS.map((section) => {
                const Icon = section.icon;
                return (
                  <button
                    key={section.id}
                    onClick={() => scrollToSection(section.id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm whitespace-nowrap transition-colors ${
                      activeSection === section.id
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {section.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Content Area */}
          <main className="flex-1 min-w-0 space-y-12">
            <section id="getting-started">
              <h2 className="text-2xl font-bold text-foreground mb-4">Getting Started</h2>
              <p className="text-muted-foreground mb-6">
                Get up and running with Msanii in just a few steps.
              </p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">1. Create Your Account</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm text-muted-foreground">
                    <p>Sign up at the home page using your Google account or email and password. Once signed in, you'll land on your Dashboard -- your central hub for accessing all tools and features.</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">2. Add Your First Artist</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm text-muted-foreground">
                    <p>Navigate to <strong>Artist Profiles</strong> from the Dashboard and click <strong>Add Artist</strong>. Fill in the artist's name, bio, genres, and connect their streaming profiles (Spotify, Apple Music, SoundCloud). You can also add social media links and custom URLs for press kits or EPKs.</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">3. Upload Contracts</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm text-muted-foreground">
                    <p>From an artist's profile, create a <strong>Project</strong> to organize related documents. Upload contract PDFs to the project -- Msanii will securely store them and make them searchable by Zoe AI. You can also upload royalty statements for use with OneClick.</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">4. Explore the Tools</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm text-muted-foreground">
                    <p>Head to the <strong>Tools</strong> page to access OneClick (royalty calculations), Zoe (AI contract assistant), and the Split Sheet Generator. Each tool is designed to save you time on common music business tasks.</p>
                  </CardContent>
                </Card>
              </div>
            </section>

            <Separator />

            <section id="oneclick">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Calculator className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">OneClick</h2>
              </div>
              <p className="text-muted-foreground mb-6">Calculate royalty splits and payments from your contracts in one click. OneClick uses AI to parse contract terms and apply them to your royalty statements automatically.</p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">How It Works</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4 text-sm text-muted-foreground">
                    <div className="space-y-3">
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Step 1:</span>
                        <p><strong>Select an Artist</strong> -- Choose the artist whose royalties you want to calculate from your roster.</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Step 2:</span>
                        <p><strong>Upload Contracts</strong> -- Upload the relevant contract PDFs. OneClick's AI will parse them to extract parties, works, and royalty split percentages for each revenue type (streaming, master, publishing, performance, mechanical, sync).</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Step 3:</span>
                        <p><strong>Upload Royalty Statements</strong> -- Upload the royalty statement files that contain the actual revenue figures.</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Step 4:</span>
                        <p><strong>Calculate</strong> -- OneClick applies the contract terms to the royalty statements and generates a detailed payment breakdown showing what each party is owed.</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Step 5:</span>
                        <p><strong>Export</strong> -- Download the results as an Excel spreadsheet with itemized breakdowns and visual charts.</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Tips</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    <ul className="list-disc list-inside space-y-2">
                      <li>Upload clear, text-based PDF contracts for the best AI extraction results. Scanned images may have lower accuracy.</li>
                      <li>You can upload multiple contracts per artist -- OneClick will parse each one independently.</li>
                      <li>Review the extracted splits before calculating to ensure the AI interpreted the contract correctly.</li>
                      <li>The pie chart visualization helps quickly identify how revenue is distributed among parties.</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </section>

            <Separator />

            <section id="zoe">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">Zoe AI</h2>
              </div>
              <p className="text-muted-foreground mb-6">Your AI-powered contract assistant. Ask questions about your uploaded contracts and get accurate answers with source citations.</p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">How to Use Zoe</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4 text-sm text-muted-foreground">
                    <p>Once you've uploaded contracts through an artist's profile, Zoe can search across all of them to answer your questions. Zoe uses semantic search to find relevant contract clauses and provides answers with citations back to the source text.</p>
                    <div className="space-y-3">
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Ask a Question:</span>
                        <p>Type your question in the chat input. Zoe works best with specific questions like "What is the royalty rate for streaming revenue in [Artist]'s contract?" rather than vague ones.</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Quick Actions:</span>
                        <p>Use the suggested quick action buttons for common queries like summarizing a contract, finding key terms, or identifying expiration dates.</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Source Citations:</span>
                        <p>Zoe includes references to the specific contract sections where it found the information, so you can verify the answers against the original documents.</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Conversation History:</span>
                        <p>Your chat history is saved automatically, so you can return to previous conversations and continue where you left off.</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Example Questions</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    <ul className="list-disc list-inside space-y-2">
                      <li>"What are the royalty split percentages in this contract?"</li>
                      <li>"When does this agreement expire?"</li>
                      <li>"What rights does the label have over the master recordings?"</li>
                      <li>"Summarize the key terms of this publishing deal."</li>
                      <li>"Are there any exclusivity clauses?"</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </section>

            <Separator />

            <section id="split-sheet">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <FileText className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">Split Sheet Generator</h2>
              </div>
              <p className="text-muted-foreground mb-6">Create professional split sheet agreements to document royalty ownership for your music. Generate clean PDFs ready for signing.</p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Step-by-Step Guide</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4 text-sm text-muted-foreground">
                    <div className="space-y-3">
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Step 1 - Details:</span>
                        <p>Enter the song title, date, and any notes about the work. This information appears at the top of the generated split sheet.</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Step 2 - Splits:</span>
                        <p>Add each contributor and define their role (songwriter, producer, performer, etc.). Set the publishing and master ownership percentages for each contributor. Optionally add IPI numbers, publisher names, and label information.</p>
                      </div>
                      <div className="flex gap-3">
                        <span className="font-semibold text-foreground shrink-0">Step 3 - Summary:</span>
                        <p>Review the complete split sheet with all contributors and percentages. Verify that publishing and master splits each total 100%. Download the split sheet as a professionally formatted PDF.</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Tips</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    <ul className="list-disc list-inside space-y-2">
                      <li>Always ensure splits total exactly 100% for both publishing and master ownership.</li>
                      <li>Include IPI numbers when available -- these are essential for collecting societies to properly route royalties.</li>
                      <li>Create split sheets before starting a project to avoid disputes later.</li>
                      <li>Download and share the PDF with all contributors for their records.</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </section>

            <Separator />

            <section id="artist-management">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Users className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">Artist Management</h2>
              </div>
              <p className="text-muted-foreground mb-6">Manage your complete artist roster with detailed profiles, streaming links, social media connections, and organized project folders.</p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Artist Profiles</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-muted-foreground">
                    <p>Each artist profile includes:</p>
                    <ul className="list-disc list-inside space-y-2">
                      <li><strong>Basic Info</strong> -- Name, bio, genres, and profile image.</li>
                      <li><strong>DSP Links</strong> -- Connect Spotify, Apple Music, and SoundCloud profiles for quick access.</li>
                      <li><strong>Social Links</strong> -- Add Instagram, Twitter/X, TikTok, YouTube, and other social media URLs.</li>
                      <li><strong>Custom Links</strong> -- Add EPKs, press kits, linktrees, or any other relevant URLs.</li>
                      <li><strong>Contacts</strong> -- Track contacts associated with each artist (managers, lawyers, label reps).</li>
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Projects & Documents</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-muted-foreground">
                    <p>Organize an artist's work into <strong>Projects</strong>. Each project acts as a folder where you can store contracts, royalty statements, and other documents. Projects help you keep related files together -- for example, group all documents for a specific album or deal into one project.</p>
                    <p>Uploaded contracts are automatically processed for use with OneClick and Zoe AI, so you only need to upload once to benefit from all tools.</p>
                  </CardContent>
                </Card>
              </div>
            </section>

            <Separator />

            <section id="workspace">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <LayoutGrid className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">Workspace</h2>
              </div>
              <p className="text-muted-foreground mb-6">Your project management hub with Kanban boards, calendar views, integrations, and team settings.</p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Kanban Boards</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-muted-foreground">
                    <p>Organize tasks visually using drag-and-drop Kanban boards. Create columns for different stages (e.g., To Do, In Progress, Done) and move cards between them as work progresses.</p>
                    <ul className="list-disc list-inside space-y-2">
                      <li>Create tasks with titles, descriptions, and due dates.</li>
                      <li>Organize tasks into parent tasks for hierarchical tracking.</li>
                      <li>Add comments to tasks for collaboration notes.</li>
                      <li>Color-code cards for quick visual identification.</li>
                      <li>Filter boards by artist to focus on specific projects.</li>
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Calendar View</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    <p>Switch to the calendar view to see your tasks plotted on a timeline. This is useful for tracking deadlines, release dates, and contract expiration dates at a glance.</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Integrations</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-muted-foreground">
                    <p>Connect your workspace with external services:</p>
                    <ul className="list-disc list-inside space-y-2">
                      <li><strong>Google Drive</strong> -- Access and link documents from your Drive.</li>
                      <li><strong>Slack</strong> -- Get notifications in your Slack workspace.</li>
                      <li><strong>Notion</strong> -- Sync with your Notion workspace.</li>
                      <li><strong>Monday.com</strong> -- Connect with your Monday.com boards.</li>
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    <p>Configure your workspace timezone and time format (12-hour or 24-hour) from the Settings tab. These preferences apply across your Dashboard and Workspace views.</p>
                  </CardContent>
                </Card>
              </div>
            </section>

            <Separator />

            <section id="portfolio">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Folder className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">Portfolio</h2>
              </div>
              <p className="text-muted-foreground mb-6">View your artist roster as a portfolio with audio previews, organized files, and shareable links.</p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Portfolio Features</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-muted-foreground">
                    <ul className="list-disc list-inside space-y-2">
                      <li><strong>Artist Cards</strong> -- Each artist appears as a card with their name, genres, and key information at a glance.</li>
                      <li><strong>Audio Previews</strong> -- Listen to streaming samples directly from the portfolio view when artists have connected their DSP profiles.</li>
                      <li><strong>File Organization</strong> -- Access contracts and royalty statements organized by artist and project.</li>
                      <li><strong>Shareable Links</strong> -- Share individual artist profiles via direct links for quick collaboration with partners or stakeholders.</li>
                      <li><strong>Search & Filter</strong> -- Quickly find artists using the search bar or filter by genre.</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </section>

            <Separator />

            <section id="contacts-payments">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Phone className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">Contacts & Payments</h2>
              </div>
              <p className="text-muted-foreground mb-6">Keep track of your industry contacts and payment history in one place.</p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Contact Management</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-muted-foreground">
                    <p>Store and organize contacts for managers, lawyers, label representatives, producers, and other industry professionals. Each contact can include name, email, phone, company, and role information. Link contacts to specific artists to keep your network organized.</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Payment Tracking</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-muted-foreground">
                    <p>Record and track payments with details including amount, date, description, and associated artist. Use the payments view to maintain a history of all financial transactions and quickly reference past payments when needed.</p>
                  </CardContent>
                </Card>
              </div>
            </section>

            <Separator />

            <section id="best-practices">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                  <Lightbulb className="w-5 h-5 text-primary" />
                </div>
                <h2 className="text-2xl font-bold text-foreground">Best Practices</h2>
              </div>
              <p className="text-muted-foreground mb-6">Tips for getting the most out of Msanii.</p>
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Organize with Projects</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    <ul className="list-disc list-inside space-y-2">
                      <li>Create a project for each deal, album, or major agreement so related documents stay together.</li>
                      <li>Name projects descriptively (e.g., "2024 Publishing Deal - Universal" rather than "Deal 1").</li>
                      <li>Upload both contracts and royalty statements to the same project for seamless OneClick calculations.</li>
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Contract Management</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    <ul className="list-disc list-inside space-y-2">
                      <li>Upload contracts as soon as they are signed to keep your records current.</li>
                      <li>Use text-based PDFs (not scanned images) for the best AI parsing accuracy.</li>
                      <li>After uploading, use Zoe to verify that key terms were correctly extracted.</li>
                      <li>Track contract expiration dates and set reminders using Workspace boards.</li>
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Workflow Recommendations</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm text-muted-foreground">
                    <ul className="list-disc list-inside space-y-2">
                      <li><strong>New artist onboarding:</strong> Create profile, add projects, upload contracts, set up Workspace board.</li>
                      <li><strong>Royalty period:</strong> Upload latest royalty statements, run OneClick, export Excel, record payments.</li>
                      <li><strong>New collaboration:</strong> Generate split sheet, share PDF with all contributors, upload signed copy to project.</li>
                      <li>Use the Dashboard's "Recently Used" section for quick access to your most-visited tools.</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
            </section>
          </main>
        </div>
      </div>
    </div>
  );
};

export default Documentation;
