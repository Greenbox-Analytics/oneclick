import { useState, useRef, useCallback } from "react";
import { useNotes, useFolders, useNote, useUpdateNote } from "@/hooks/useNotes";
import { useIsMobile } from "@/hooks/use-mobile";
import NotesEditor from "./NotesEditor";
import NotesSidebar from "./NotesSidebar";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Pin, PinOff } from "lucide-react";

interface Props {
  scope: { artistId?: string; projectId?: string };
  className?: string;
}

export default function NotesView({ scope, className }: Props) {
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const [localTitle, setLocalTitle] = useState("");
  const titleDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMobile = useIsMobile();

  const { data: notes = [] } = useNotes(scope);
  const { data: folders = [] } = useFolders(scope);
  const { data: selectedNote } = useNote(selectedNoteId ?? undefined);
  const updateNote = useUpdateNote();

  // Sync local title from server when note selection changes
  const prevNoteIdRef = useRef<string | null>(null);
  if (selectedNote && selectedNoteId !== prevNoteIdRef.current) {
    prevNoteIdRef.current = selectedNoteId;
    setLocalTitle(selectedNote.title);
  }

  const handleContentChange = (content: unknown[]) => {
    if (!selectedNoteId) return;
    updateNote.mutate({ noteId: selectedNoteId, content });
  };

  const handleTitleChange = useCallback((title: string) => {
    setLocalTitle(title); // Update local state immediately (optimistic)
    if (!selectedNoteId) return;
    // Debounce the API call
    if (titleDebounceRef.current) clearTimeout(titleDebounceRef.current);
    titleDebounceRef.current = setTimeout(() => {
      if (title.trim()) {
        updateNote.mutate({ noteId: selectedNoteId, title: title.trim() });
      }
    }, 800);
  }, [selectedNoteId, updateNote]);

  const togglePin = () => {
    if (!selectedNoteId || !selectedNote) return;
    updateNote.mutate({ noteId: selectedNoteId, pinned: !selectedNote.pinned });
  };

  // On mobile, show one pane at a time: the notes list, or (once a note is
  // picked) the editor with a "back to list" control. On desktop, show both
  // side-by-side as before.
  const showSidebar = !isMobile || !selectedNoteId;
  const showEditor = !isMobile || !!selectedNoteId;

  return (
    <div
      className={`flex border rounded-lg overflow-hidden bg-background h-[70vh] sm:h-[600px] ${className || ""}`}
    >
      {showSidebar && (
        <NotesSidebar
          folders={folders}
          notes={notes}
          selectedNoteId={selectedNoteId}
          onSelectNote={setSelectedNoteId}
          scope={scope}
        />
      )}
      {showEditor && (
        <div className="flex-1 flex flex-col min-w-0">
          {selectedNote ? (
            <>
              <div className="p-3 border-b flex items-center gap-2">
                {isMobile && (
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 shrink-0"
                    aria-label="Back to notes list"
                    onClick={() => setSelectedNoteId(null)}
                  >
                    <ArrowLeft className="w-4 h-4" />
                  </Button>
                )}
                <Input
                  value={localTitle}
                  onChange={(e) => handleTitleChange(e.target.value)}
                  className="border-0 text-lg font-semibold p-0 h-auto focus-visible:ring-0"
                  placeholder="Untitled"
                />
                <Button size="icon" variant="ghost" className="h-7 w-7 shrink-0" onClick={togglePin}>
                  {selectedNote.pinned ? <PinOff className="w-4 h-4 text-amber-500" /> : <Pin className="w-4 h-4" />}
                </Button>
              </div>
              <div className="flex-1 overflow-y-auto p-4">
                <NotesEditor
                  key={selectedNote.id}
                  initialContent={selectedNote.content as unknown[]}
                  onChange={handleContentChange}
                />
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-muted-foreground">
              <p className="text-sm">Select a note or create a new one</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
