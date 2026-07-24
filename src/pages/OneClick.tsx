import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { HeaderDocsButton } from "@/components/layout/HeaderDocsButton";
import { Checkbox } from "@/components/ui/checkbox";
import { AlertCircle, ArrowLeft, Info, Calculator, Wallet } from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useAuth } from "@/contexts/AuthContext";
import { RequireFeature } from "@/components/paywall/RequireFeature";
import { useAnalytics } from "@/hooks/useAnalytics";
import { useSmartBack } from "@/hooks/useSmartBack";
import { PaymentTracking } from "@/components/oneclick/payments/PaymentTracking";

import { API_URL, apiFetch } from "@/lib/apiFetch";

interface Artist {
  id: string; // UUID in database
  name: string;
  has_contract: boolean;
  // Add other fields as needed
}

const OneClick = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const goBack = useSmartBack("/tools");
  const { user } = useAuth();

  // When arriving from a OneClick calculation's "Pay Royalties" button, open the
  // Royalty Tracking tab and pre-select that calc's collaborators in the payout
  // modal. Consume the state once so it doesn't re-fire on re-render.
  const [initialPayoutNames, setInitialPayoutNames] = useState<string[] | undefined>(undefined);

  // State for fetched artists
  const [artists, setArtists] = useState<Artist[]>([]);
  const hasFetchedRef = useRef(false);

  // State tracks artist user has selected and errors
  // Changed to string[] because Supabase IDs are UUIDs (strings)
  const [selectedArtists, setSelectedArtists] = useState<string[]>([]);
  const [error, setError] = useState<string>("");

  const [tab, setTab] = useState("calculate");

  const { captureToolOpened } = useAnalytics();
  useEffect(() => {
    captureToolOpened("oneclick");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const names = (location.state as { openPayoutForNames?: string[] } | null)?.openPayoutForNames;
    if (names && names.length > 0) {
      setTab("payments");
      setInitialPayoutNames(names);
      // Clear router state so a refresh/re-render doesn't reopen the modal.
      navigate(location.pathname, { replace: true, state: {} });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);

  // Fetch artists from backend on component mount
  useEffect(() => {
    if (!user || hasFetchedRef.current) return;

    hasFetchedRef.current = true;

    apiFetch<Artist[]>(`${API_URL}/artists`)
      .then((data) => setArtists(data))
      .catch((err) => {
        console.error("Error fetching artists:", err);
        setError("Failed to load artists. Please check your backend connection.");
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  // This lets us select/deselect a single artist (only one artist can be selected at a time)
  const handleArtistToggle = (artistId: string) => {
    setSelectedArtists((prev) =>
      prev.includes(artistId)
        ? [] // Deselect if already selected
        : [artistId] // Select only this artist (replaces any previous selection)
    );
  };

  // Navigate to document selection page for the selected artist
  const handleContinue = () => {
    setError("");

    if (selectedArtists.length === 0) {
      setError("Please select at least one artist");
      return;
    }

    const selectedArtistId = selectedArtists[0];
    navigate(`/oneclick/${selectedArtistId}/documents`);
  };

  return (
    <RequireFeature feature="oneclick">
      <div className="min-h-screen bg-background">
        <PageHeader
          showBack={false}
          actions={
          <>
            <HeaderDocsButton />
            <Button variant="outline" className="hidden md:inline-flex" onClick={goBack}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
          </>
        }
      />

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">OneClick</h2>
          <p className="text-muted-foreground">Calculate royalty splits and track payouts across your artists</p>
        </div>

        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            <TabsTrigger value="calculate" className="gap-1.5"><Calculator className="w-4 h-4" /> Calculate</TabsTrigger>
            <TabsTrigger value="payments" className="gap-1.5"><Wallet className="w-4 h-4" /> Royalty Tracking</TabsTrigger>
          </TabsList>

          <TabsContent value="calculate" className="mt-6">
            <Alert className="mb-6">
              <Info className="w-4 h-4" />
              <AlertTitle>Streaming Master royalties only</AlertTitle>
              <AlertDescription>
                OneClick currently analyzes master royalties related to streaming earnings only. Split sheets generated for publishing-only royalties will not be recognized.
              </AlertDescription>
            </Alert>

            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Select Artists</CardTitle>
                  <CardDescription>Choose artists to include in the royalty calculation</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {artists.length === 0 ? (
                    <div className="text-center py-4 text-muted-foreground">
                      No artists found. Please add an artist first.
                    </div>
                  ) : (
                    artists.map((artist) => (
                      <div
                        key={artist.id}
                        className="flex items-center p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <Checkbox
                            id={`artist-${artist.id}`}
                            checked={selectedArtists.includes(artist.id)}
                            onCheckedChange={() => handleArtistToggle(artist.id)}
                          />
                          <label
                            htmlFor={`artist-${artist.id}`}
                            className="text-foreground font-medium cursor-pointer"
                          >
                            {artist.name}
                          </label>
                        </div>
                      </div>
                    ))
                  )}

                  <Button
                    onClick={handleContinue}
                    className="w-full"
                    disabled={selectedArtists.length === 0}
                  >
                    Continue
                  </Button>
                </CardContent>
              </Card>

              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </div>
          </TabsContent>

          <TabsContent value="payments" className="mt-6">
            <PaymentTracking
              initialPayoutNames={initialPayoutNames}
              onPayoutConsumed={() => setInitialPayoutNames(undefined)}
            />
          </TabsContent>
        </Tabs>
      </main>
      </div>
    </RequireFeature>
  );
};

export default OneClick;
