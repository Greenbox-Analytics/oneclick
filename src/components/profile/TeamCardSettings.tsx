import { useState } from "react";
import { useMyTeamCard, useUpdateTeamCard, type TeamCard } from "@/hooks/useTeamCard";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Eye, Settings, Lock } from "lucide-react";

const ALL_FIELDS = [
  { key: "display_name", label: "Display Name", locked: true },
  { key: "first_name", label: "First Name", locked: true },
  { key: "last_name", label: "Last Name", locked: true },
  { key: "email", label: "Email", locked: true },
  { key: "avatar_url", label: "Profile Photo", locked: false },
  { key: "bio", label: "Bio", locked: false },
  { key: "phone", label: "Phone", locked: false },
  { key: "website", label: "Website", locked: false },
  { key: "company", label: "Company", locked: false },
  { key: "role", label: "Role", locked: false },
  { key: "social_links", label: "Social Links", locked: false },
  { key: "dsp_links", label: "Streaming Platforms", locked: false },
  { key: "custom_links", label: "Custom Links", locked: false },
];

const LOCKED_FIELDS = ALL_FIELDS.filter((f) => f.locked).map((f) => f.key);

export default function TeamCardSettings() {
  const { data: card, isLoading } = useMyTeamCard();
  const updateCard = useUpdateTeamCard();
  const [editMode, setEditMode] = useState(false);
  const [displayName, setDisplayName] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [bio, setBio] = useState("");
  const [phone, setPhone] = useState("");
  const [website, setWebsite] = useState("");
  const [company, setCompany] = useState("");
  const [visibleFields, setVisibleFields] = useState<string[]>([]);

  const startEdit = () => {
    if (!card) return;
    setDisplayName(card.display_name);
    setFirstName(card.first_name);
    setLastName(card.last_name);
    setBio(card.bio || "");
    setPhone(card.phone || "");
    setWebsite(card.website || "");
    setCompany(card.company || "");
    setVisibleFields(card.visible_fields || LOCKED_FIELDS);
    setEditMode(true);
  };

  const handleSave = async () => {
    await updateCard.mutateAsync({
      display_name: displayName,
      first_name: firstName,
      last_name: lastName,
      bio: bio || undefined,
      phone: phone || undefined,
      website: website || undefined,
      company: company || undefined,
      visible_fields: [...new Set([...LOCKED_FIELDS, ...visibleFields])],
    });
    setEditMode(false);
  };

  const toggleVisibility = (key: string) => {
    setVisibleFields((prev) =>
      prev.includes(key) ? prev.filter((f) => f !== key) : [...prev, key]
    );
  };

  if (isLoading) {
    return (
      <Card><CardContent className="py-8 text-center"><Loader2 className="w-6 h-6 animate-spin mx-auto" /></CardContent></Card>
    );
  }

  if (!card) {
    return (
      <Card><CardContent className="py-8 text-center text-muted-foreground">
        TeamCard will be created after onboarding.
      </CardContent></Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">TeamCard</CardTitle>
            <CardDescription>
              Your collaboration identity — choose what collaborators see about you
            </CardDescription>
          </div>
          {!editMode && (
            <Button variant="outline" size="sm" onClick={startEdit}>
              <Settings className="w-4 h-4 mr-1" /> Configure
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue={editMode ? "edit" : "preview"}>
          <TabsList className="mb-4">
            <TabsTrigger value="preview"><Eye className="w-4 h-4 mr-1" /> Preview</TabsTrigger>
            {editMode && <TabsTrigger value="edit"><Settings className="w-4 h-4 mr-1" /> Edit</TabsTrigger>}
            {editMode && <TabsTrigger value="visibility">Visibility</TabsTrigger>}
          </TabsList>

          <TabsContent value="preview">
            <div className="border rounded-lg p-4 bg-muted/30">
              <div className="flex items-center gap-3 mb-3">
                {card.avatar_url && (card.visible_fields || []).includes("avatar_url") && (
                  <img src={card.avatar_url} alt="" className="w-12 h-12 rounded-full object-cover" />
                )}
                <div>
                  <p className="font-semibold">{card.display_name}</p>
                  <p className="text-sm text-muted-foreground">{card.email}</p>
                </div>
                <Badge className="bg-green-100 text-green-800 ml-auto">Verified</Badge>
              </div>
              {card.bio && (card.visible_fields || []).includes("bio") && (
                <p className="text-sm text-muted-foreground mb-2">{card.bio}</p>
              )}
              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                {card.company && (card.visible_fields || []).includes("company") && <span>{card.company}</span>}
                {card.role && (card.visible_fields || []).includes("role") && <span>· {card.role}</span>}
                {card.website && (card.visible_fields || []).includes("website") && <span>· {card.website}</span>}
              </div>
            </div>
          </TabsContent>

          {editMode && (
            <TabsContent value="edit">
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium">Display Name *</label>
                  <Input value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium">First Name *</label>
                    <Input value={firstName} onChange={(e) => setFirstName(e.target.value)} />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Last Name *</label>
                    <Input value={lastName} onChange={(e) => setLastName(e.target.value)} />
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium">Email</label>
                  <Input value={card.email} disabled className="bg-muted" />
                  <p className="text-xs text-muted-foreground mt-1">Email cannot be changed</p>
                </div>
                <div>
                  <label className="text-sm font-medium">Bio</label>
                  <Input value={bio} onChange={(e) => setBio(e.target.value)} placeholder="Short bio" />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium">Phone</label>
                    <Input value={phone} onChange={(e) => setPhone(e.target.value)} />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Website</label>
                    <Input value={website} onChange={(e) => setWebsite(e.target.value)} />
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium">Company</label>
                  <Input value={company} onChange={(e) => setCompany(e.target.value)} />
                </div>
                <div className="flex gap-2">
                  <Button onClick={handleSave} disabled={updateCard.isPending}>
                    {updateCard.isPending && <Loader2 className="w-4 h-4 mr-1 animate-spin" />}
                    Save Changes
                  </Button>
                  <Button variant="outline" onClick={() => setEditMode(false)}>Cancel</Button>
                </div>
              </div>
            </TabsContent>
          )}

          {editMode && (
            <TabsContent value="visibility">
              <p className="text-sm text-muted-foreground mb-3">Toggle which fields collaborators can see on your TeamCard.</p>
              <div className="space-y-3">
                {ALL_FIELDS.map((field) => (
                  <div key={field.key} className="flex items-center justify-between py-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{field.label}</span>
                      {field.locked && <Lock className="w-3 h-3 text-muted-foreground" />}
                    </div>
                    <Switch
                      checked={field.locked || visibleFields.includes(field.key)}
                      onCheckedChange={() => !field.locked && toggleVisibility(field.key)}
                      disabled={field.locked}
                    />
                  </div>
                ))}
              </div>
              <Button className="mt-4" onClick={handleSave} disabled={updateCard.isPending}>
                {updateCard.isPending && <Loader2 className="w-4 h-4 mr-1 animate-spin" />}
                Save Visibility
              </Button>
            </TabsContent>
          )}
        </Tabs>
      </CardContent>
    </Card>
  );
}
