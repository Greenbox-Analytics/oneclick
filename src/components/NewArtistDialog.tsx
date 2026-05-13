import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Upload } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";

interface NewArtistDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: (artistId: string) => void;
}

export const NewArtistDialog = ({ open, onOpenChange, onCreated }: NewArtistDialogProps) => {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [genre, setGenre] = useState("");
  const [phone, setPhone] = useState("");
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setName("");
      setEmail("");
      setGenre("");
      setPhone("");
      setAvatarPreview(null);
    }
  }, [open]);

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setAvatarPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const mutation = useMutation({
    mutationFn: async () => {
      if (!user) {
        throw new Error("You must be logged in to create an artist");
      }
      const { data, error } = await supabase
        .from("artists")
        .insert({
          name,
          email,
          bio: "",
          genres: genre.split(",").map((g) => g.trim()).filter(Boolean),
          avatar_url: avatarPreview || "",
          user_id: user.id,
        })
        .select()
        .single();
      if (error) throw error;
      return data;
    },
    onSuccess: async (newArtist) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["portfolio-artists"] }),
        queryClient.invalidateQueries({ queryKey: ["artists-with-teamcards"] }),
      ]);
      toast({ title: "Success", description: "Artist created successfully" });
      onCreated(newArtist.id);
      onOpenChange(false);
    },
    onError: (err: Error) => {
      toast({
        title: "Error",
        description: err.message || "Failed to create artist",
        variant: "destructive",
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim() || !genre.trim()) return;
    mutation.mutate();
  };

  const isSubmitting = mutation.isPending;
  const canSubmit = !!name.trim() && !!email.trim() && !!genre.trim() && !isSubmitting;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Add New Artist</DialogTitle>
          <DialogDescription>Create a new artist profile</DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label>Profile Picture</Label>
            <div className="flex items-center gap-4">
              <Avatar className="w-16 h-16">
                <AvatarImage src={avatarPreview || undefined} />
                <AvatarFallback className="bg-muted">
                  <Upload className="w-6 h-6 text-muted-foreground" />
                </AvatarFallback>
              </Avatar>
              <div className="flex-1">
                <Input
                  id="new-artist-avatar"
                  type="file"
                  accept="image/*"
                  onChange={handleAvatarChange}
                  className="cursor-pointer"
                />
                <p className="text-xs text-muted-foreground mt-1">Upload JPG, PNG or GIF</p>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="new-artist-name">
              Artist Name <span className="text-green-500">*</span>
            </Label>
            <Input
              id="new-artist-name"
              placeholder="Enter artist name"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="new-artist-genre">
              Genre <span className="text-green-500">*</span>
            </Label>
            <Input
              id="new-artist-genre"
              placeholder="e.g., Electronic, Rock, Hip-Hop"
              required
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="new-artist-email">
              Email <span className="text-green-500">*</span>
            </Label>
            <Input
              id="new-artist-email"
              type="email"
              placeholder="artist@example.com"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="new-artist-phone">Phone (Optional)</Label>
            <Input
              id="new-artist-phone"
              type="tel"
              placeholder="+1 (555) 123-4567"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
            />
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={!canSubmit}>
              {isSubmitting ? "Creating..." : "Create Artist"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
