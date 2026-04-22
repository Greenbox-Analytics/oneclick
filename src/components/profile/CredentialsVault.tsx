import { useEffect, useMemo, useRef, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Edit, Eye, EyeOff, Lock, Plus, Trash2, ExternalLink } from "lucide-react";
import {
  useArtistCredentials,
  useCreateCredential,
  useDeleteCredential,
  useUpdateCredential,
  type ArtistCredential,
} from "@/hooks/useArtistCredentials";
import PasswordRevealDialog from "./PasswordRevealDialog";

interface Props {
  artistId: string;
  isEditMode: boolean;
}

const REVEAL_TIMEOUT_MS = 30_000;

interface DraftCredential {
  platform_name: string;
  login_identifier: string;
  password: string;
  url: string;
  notes: string;
}

const EMPTY_DRAFT: DraftCredential = {
  platform_name: "",
  login_identifier: "",
  password: "",
  url: "",
  notes: "",
};

export default function CredentialsVault({ artistId, isEditMode }: Props) {
  const { data: credentials = [], isLoading } = useArtistCredentials(artistId);
  const createMutation = useCreateCredential(artistId);
  const updateMutation = useUpdateCredential(artistId);
  const deleteMutation = useDeleteCredential(artistId);

  const [draft, setDraft] = useState<DraftCredential | null>(null);
  const [revealDialogFor, setRevealDialogFor] = useState<string | null>(null);
  const [revealed, setRevealed] = useState<Record<string, string>>({});
  const timeoutsRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  useEffect(() => {
    const timeouts = timeoutsRef.current;
    return () => {
      for (const id of Object.keys(timeouts)) {
        clearTimeout(timeouts[id]);
      }
    };
  }, []);

  const clearRevealed = (credentialId: string) => {
    setRevealed((prev) => {
      const next = { ...prev };
      delete next[credentialId];
      return next;
    });
    delete timeoutsRef.current[credentialId];
  };

  const handleRevealed = (credentialId: string, password: string) => {
    setRevealed((prev) => ({ ...prev, [credentialId]: password }));
    if (timeoutsRef.current[credentialId]) {
      clearTimeout(timeoutsRef.current[credentialId]);
    }
    timeoutsRef.current[credentialId] = setTimeout(
      () => clearRevealed(credentialId),
      REVEAL_TIMEOUT_MS,
    );
  };

  const handleHide = (credentialId: string) => {
    if (timeoutsRef.current[credentialId]) {
      clearTimeout(timeoutsRef.current[credentialId]);
    }
    clearRevealed(credentialId);
  };

  const handleSaveDraft = async () => {
    if (!draft) return;
    if (!draft.platform_name.trim() || !draft.login_identifier.trim() || !draft.password) return;
    await createMutation.mutateAsync({
      platform_name: draft.platform_name.trim(),
      login_identifier: draft.login_identifier.trim(),
      password: draft.password,
      url: draft.url.trim() || null,
      notes: draft.notes.trim() || null,
    });
    setDraft(null);
  };

  const handleDelete = async (credentialId: string) => {
    await deleteMutation.mutateAsync(credentialId);
    clearRevealed(credentialId);
  };

  const sorted = useMemo(
    () => [...credentials].sort((a, b) => a.platform_name.localeCompare(b.platform_name)),
    [credentials],
  );

  return (
    <>
      <Card className="border border-border shadow-sm hover:shadow-md transition-shadow overflow-hidden">
        <div className="h-0.5 bg-rose-500/40" />
        <CardHeader className="bg-gradient-to-r from-rose-500/5 to-transparent">
          <div className="flex items-center gap-2">
            <Lock className="w-5 h-5 text-rose-500" />
            <CardTitle>Credentials Vault</CardTitle>
          </div>
          <CardDescription>
            Store logins for platforms you use (DistroKid, Spotify for Artists, ASCAP/BMI, etc.). Passwords are encrypted at rest and only revealed after re-entering your Msanii password.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 pt-6">
          {isLoading && (
            <p className="text-sm text-muted-foreground text-center py-4">Loading credentials…</p>
          )}

          {!isLoading && sorted.length === 0 && !draft && (
            <p className="text-sm text-muted-foreground text-center py-4">
              {isEditMode
                ? 'No credentials saved yet. Click "Add Credential" to create one.'
                : "No credentials saved yet."}
            </p>
          )}

          {sorted.map((cred) => (
            <CredentialRow
              key={cred.id}
              credential={cred}
              isEditMode={isEditMode}
              revealedPassword={revealed[cred.id]}
              onRequestReveal={() => setRevealDialogFor(cred.id)}
              onHide={() => handleHide(cred.id)}
              onDelete={() => handleDelete(cred.id)}
              onUpdate={(patch) =>
                updateMutation.mutateAsync({ credentialId: cred.id, patch })
              }
            />
          ))}

          {draft && (
            <div className="p-4 rounded-lg border border-dashed border-rose-500/40 bg-card/50 space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>Platform *</Label>
                  <Input
                    value={draft.platform_name}
                    onChange={(e) => setDraft({ ...draft, platform_name: e.target.value })}
                    placeholder="e.g. DistroKid"
                  />
                </div>
                <div className="space-y-1">
                  <Label>Login / email *</Label>
                  <Input
                    value={draft.login_identifier}
                    onChange={(e) => setDraft({ ...draft, login_identifier: e.target.value })}
                    placeholder="artist@example.com"
                  />
                </div>
              </div>
              <div className="space-y-1">
                <Label>Password *</Label>
                <Input
                  type="password"
                  value={draft.password}
                  onChange={(e) => setDraft({ ...draft, password: e.target.value })}
                  autoComplete="new-password"
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>URL</Label>
                  <Input
                    value={draft.url}
                    onChange={(e) => setDraft({ ...draft, url: e.target.value })}
                    placeholder="https://distrokid.com"
                  />
                </div>
                <div className="space-y-1">
                  <Label>Notes</Label>
                  <Input
                    value={draft.notes}
                    onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
                    placeholder="Optional notes"
                  />
                </div>
              </div>
              <div className="flex gap-2 justify-end">
                <Button variant="ghost" onClick={() => setDraft(null)}>
                  Cancel
                </Button>
                <Button
                  onClick={handleSaveDraft}
                  disabled={
                    !draft.platform_name.trim() ||
                    !draft.login_identifier.trim() ||
                    !draft.password ||
                    createMutation.isPending
                  }
                >
                  Save Credential
                </Button>
              </div>
            </div>
          )}

          {isEditMode && !draft && (
            <Button variant="outline" size="sm" onClick={() => setDraft({ ...EMPTY_DRAFT })}>
              <Plus className="w-4 h-4 mr-1" />
              Add Credential
            </Button>
          )}
        </CardContent>
      </Card>

      <PasswordRevealDialog
        open={revealDialogFor !== null}
        onOpenChange={(open) => {
          if (!open) setRevealDialogFor(null);
        }}
        credentialId={revealDialogFor}
        onRevealed={(credentialId, password) => {
          handleRevealed(credentialId, password);
          setRevealDialogFor(null);
        }}
      />
    </>
  );
}

interface RowProps {
  credential: ArtistCredential;
  isEditMode: boolean;
  revealedPassword: string | undefined;
  onRequestReveal: () => void;
  onHide: () => void;
  onDelete: () => void;
  onUpdate: (patch: { platform_name?: string; login_identifier?: string; url?: string | null; notes?: string | null }) => Promise<unknown>;
}

function CredentialRow({
  credential,
  isEditMode,
  revealedPassword,
  onRequestReveal,
  onHide,
  onDelete,
  onUpdate,
}: RowProps) {
  const [editing, setEditing] = useState(false);
  const [platform, setPlatform] = useState(credential.platform_name);
  const [login, setLogin] = useState(credential.login_identifier);
  const [url, setUrl] = useState(credential.url ?? "");
  const [notes, setNotes] = useState(credential.notes ?? "");

  const handleSave = async () => {
    await onUpdate({
      platform_name: platform.trim(),
      login_identifier: login.trim(),
      url: url.trim() || null,
      notes: notes.trim() || null,
    });
    setEditing(false);
  };

  const handleCancelEdit = () => {
    setPlatform(credential.platform_name);
    setLogin(credential.login_identifier);
    setUrl(credential.url ?? "");
    setNotes(credential.notes ?? "");
    setEditing(false);
  };

  return (
    <div className="p-4 rounded-lg border border-border bg-card/50">
      {editing ? (
        <div className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label>Platform</Label>
              <Input value={platform} onChange={(e) => setPlatform(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label>Login / email</Label>
              <Input value={login} onChange={(e) => setLogin(e.target.value)} />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label>URL</Label>
              <Input value={url} onChange={(e) => setUrl(e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label>Notes</Label>
              <Input value={notes} onChange={(e) => setNotes(e.target.value)} />
            </div>
          </div>
          <div className="flex gap-2 justify-end">
            <Button variant="ghost" size="sm" onClick={handleCancelEdit}>
              Cancel
            </Button>
            <Button size="sm" onClick={handleSave}>
              Save
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="font-semibold text-foreground">{credential.platform_name}</p>
              <p className="text-sm text-muted-foreground">{credential.login_identifier}</p>
            </div>
            {isEditMode && (
              <div className="flex gap-1">
                <Button variant="ghost" size="icon" onClick={() => setEditing(true)} title="Edit">
                  <Edit className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onDelete}
                  className="text-destructive hover:text-destructive hover:bg-destructive/10"
                  title="Delete"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Label className="text-xs text-muted-foreground w-20 shrink-0">Password</Label>
            <code className="flex-1 text-sm bg-muted px-2 py-1 rounded break-all">
              {revealedPassword ?? "••••••••••"}
            </code>
            {revealedPassword ? (
              <Button variant="ghost" size="icon" onClick={onHide} title="Hide">
                <EyeOff className="w-4 h-4" />
              </Button>
            ) : (
              <Button variant="ghost" size="icon" onClick={onRequestReveal} title="Reveal">
                <Eye className="w-4 h-4" />
              </Button>
            )}
          </div>

          {credential.url && (
            <div className="flex items-center gap-2">
              <Label className="text-xs text-muted-foreground w-20 shrink-0">URL</Label>
              <a
                href={credential.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary hover:underline break-all inline-flex items-center gap-1"
              >
                {credential.url}
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}
          {credential.notes && (
            <div className="flex items-start gap-2">
              <Label className="text-xs text-muted-foreground w-20 shrink-0 pt-1">Notes</Label>
              <p className="text-sm text-muted-foreground break-words">{credential.notes}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
