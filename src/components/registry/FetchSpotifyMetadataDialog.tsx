import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, Loader2, Search, Sparkles } from "lucide-react";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { type SpotifyTrack, useSpotifySearch } from "@/hooks/useSpotifySearch";

interface CurrentMeta {
  isrc: string | null;
  upc: string | null;
  release_date: string | null;
  notes: string | null;
}

type MetaPatch = Partial<{ isrc: string; upc: string; release_date: string; notes: string }>;

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  artistName: string;
  workTitle: string;
  currentMeta: CurrentMeta;
  onApply: (patch: MetaPatch) => void;
}

// Fields we can actually write back to works_registry. Genre/label live on
// the Spotify payload but have no destination column today.
const PATCHABLE: Array<{ key: keyof CurrentMeta; label: string; from: keyof SpotifyTrack }> = [
  { key: "isrc", label: "ISRC", from: "isrc" },
  { key: "upc", label: "UPC", from: "upc" },
  { key: "release_date", label: "Release date", from: "release_date" },
  { key: "notes", label: "Spotify link", from: "spotify_url" },
];

function hasSpotifyUrl(notes: string | null): boolean {
  return !!notes && /open\.spotify\.com\/track\//.test(notes);
}

export default function FetchSpotifyMetadataDialog({
  open,
  onOpenChange,
  artistName,
  workTitle,
  currentMeta,
  onApply,
}: Props) {
  // Seed the query once per open with "artist + title"; user can edit before searching.
  const seedQuery = `${artistName} ${workTitle}`.trim();
  const [query, setQuery] = useState(seedQuery);
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [pickedId, setPickedId] = useState<string | null>(null);
  const [enriched, setEnriched] = useState<SpotifyTrack | null>(null);
  const [enriching, setEnriching] = useState(false);

  // Reset state every time the dialog opens with a (possibly new) work.
  useEffect(() => {
    if (open) {
      setQuery(seedQuery);
      setSearchEnabled(false);
      setPickedId(null);
      setEnriched(null);
      setEnriching(false);
    }
  }, [open, seedQuery]);

  const searchQuery = useSpotifySearch(query, searchEnabled);

  const runSearch = () => {
    setPickedId(null);
    setEnriched(null);
    setSearchEnabled(true);
  };

  const pick = async (r: SpotifyTrack) => {
    setPickedId(r.id);
    // Seed enriched from the search result so the preview renders immediately,
    // then upgrade with the /tracks/{id} payload (album-fetched UPC, artist-fetched genre).
    setEnriched(r);
    setEnriching(true);
    try {
      const full = await apiFetch<SpotifyTrack>(
        `${API_URL}/integrations/spotify/tracks/${r.id}`
      );
      setEnriched((prev) => (prev?.id === r.id ? { ...prev, ...full } : prev));
    } catch {
      // Search-result data is still usable; just don't get the album/artist enrichments.
    } finally {
      setEnriching(false);
    }
  };

  // Always replace from Spotify when the source has a value — the user explicitly
  // ran this flow, so the Spotify value is authoritative. Only skip a field when
  // Spotify itself has nothing to offer (so we don't blank out an existing value).
  const patch = useMemo<MetaPatch>(() => {
    if (!enriched) return {};
    const p: MetaPatch = {};
    if (enriched.isrc) p.isrc = enriched.isrc;
    if (enriched.upc) p.upc = enriched.upc;
    if (enriched.release_date) p.release_date = enriched.release_date;
    if (enriched.spotify_url) p.notes = enriched.spotify_url;
    return p;
  }, [enriched]);

  const willFillCount = Object.keys(patch).length;

  const apply = () => {
    if (willFillCount === 0) {
      toast.info("Spotify didn't return any metadata for this track.");
      onOpenChange(false);
      return;
    }
    onApply(patch);
    toast.success(`Updated ${willFillCount} field${willFillCount === 1 ? "" : "s"} from Spotify`);
    onOpenChange(false);
  };

  const results = searchQuery.data;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-primary" />
            Fetch metadata from Spotify
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4 pt-1">
          <p className="text-sm text-muted-foreground">
            Now that "{workTitle}" is released, we can pull the public Spotify metadata —
            ISRC, UPC, release date, and the Spotify link. The fetched values will
            <b> overwrite</b> any existing entries for these fields.
          </p>

          {/* Search */}
          <div className="space-y-1.5">
            <Label className="text-sm font-medium">Search</Label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  autoFocus
                  placeholder="Artist name + song title"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && runSearch()}
                  className="pl-9"
                />
              </div>
              <Button onClick={runSearch} disabled={!query.trim() || searchQuery.isFetching}>
                {searchQuery.isFetching ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : null}
                Search
              </Button>
            </div>
          </div>

          {/* Results */}
          {searchEnabled && searchQuery.isFetching && !results && (
            <div className="space-y-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="flex items-center gap-3 p-2 rounded-lg border animate-pulse">
                  <div className="w-11 h-11 rounded-lg bg-muted" />
                  <div className="flex-1">
                    <div className="h-3 w-1/2 bg-muted rounded mb-2" />
                    <div className="h-2 w-1/3 bg-muted rounded" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {results && results.length === 0 && (
            <div className="rounded-lg border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
              No matches. Try refining the query (artist + song title usually works best).
            </div>
          )}

          {results && results.length > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  <b className="text-foreground">{results.length}</b> results
                </span>
                <span>Pick the correct match</span>
              </div>
              <div className="space-y-1.5 max-h-[280px] overflow-y-auto pr-1">
                {results.map((r) => {
                  const picked = pickedId === r.id;
                  return (
                    <button
                      key={r.id}
                      type="button"
                      onClick={() => pick(r)}
                      className={cn(
                        "w-full flex items-center gap-3 p-2 rounded-lg border text-left transition-colors",
                        picked
                          ? "border-primary bg-primary/5"
                          : "hover:border-primary/40 hover:bg-muted/40"
                      )}
                    >
                      {r.cover_url ? (
                        <img src={r.cover_url} alt="" className="w-11 h-11 rounded-md shrink-0" />
                      ) : (
                        <div className="w-11 h-11 rounded-md bg-muted shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-semibold truncate">{r.title}</div>
                        <div className="text-xs text-muted-foreground truncate">
                          {r.artist}
                          {r.album ? ` · ${r.album}` : ""}
                          {r.year ? ` · ${r.year}` : ""}
                        </div>
                      </div>
                      {picked && <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Preview / confirm */}
          {enriched && (
            <div className="rounded-lg border border-border bg-muted/20 p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-semibold">Preview</p>
                {enriching && (
                  <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                    <Loader2 className="w-3 h-3 animate-spin" /> enriching…
                  </span>
                )}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                {PATCHABLE.map(({ key, label, from }) => {
                  const spotifyValue = (enriched[from] as string | null) || null;
                  const currentValue = currentMeta[key];
                  const willOverwrite =
                    !!spotifyValue &&
                    !!currentValue &&
                    (key === "notes" ? hasSpotifyUrl(currentValue) : true) &&
                    spotifyValue !== currentValue;
                  return (
                    <div key={key} className="flex items-start justify-between gap-2 py-1">
                      <span className="text-muted-foreground shrink-0">{label}</span>
                      <span
                        className={cn(
                          "text-right font-medium truncate",
                          spotifyValue
                            ? "text-emerald-600 dark:text-emerald-400"
                            : "text-muted-foreground"
                        )}
                        title={spotifyValue || ""}
                      >
                        {spotifyValue || "—"}
                        {willOverwrite && (
                          <span className="ml-1.5 text-[10px] not-italic font-normal text-amber-600 dark:text-amber-400">
                            (replaces existing)
                          </span>
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="text-[11px] text-muted-foreground pt-1">
                {willFillCount > 0
                  ? `Will update ${willFillCount} field${willFillCount === 1 ? "" : "s"} from Spotify.`
                  : "Spotify didn't return any metadata for this track."}
              </p>
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 mt-4 pt-4 border-t">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={apply} disabled={!enriched || enriching}>
            <Sparkles className="w-4 h-4 mr-1" />
            {willFillCount > 0
              ? `Update ${willFillCount} field${willFillCount === 1 ? "" : "s"}`
              : "Done"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
