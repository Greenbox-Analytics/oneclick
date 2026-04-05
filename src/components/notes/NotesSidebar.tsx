import { useState } from "react";
import { type Note, type NoteFolder, useCreateNote, useCreateFolder, useDeleteFolder, useDeleteNote } from "@/hooks/useNotes";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { FolderPlus, FilePlus, Folder, FileText, Trash2, Pin, ChevronRight, ChevronDown } from "lucide-react";

interface Props {
  folders: NoteFolder[];
  notes: Note[];
  selectedNoteId: string | null;
  onSelectNote: (noteId: string) => void;
  scope: { artistId?: string; projectId?: string };
}

export default function NotesSidebar({ folders, notes, selectedNoteId, onSelectNote, scope }: Props) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [newFolderName, setNewFolderName] = useState("");
  const [showNewFolder, setShowNewFolder] = useState(false);

  const createNote = useCreateNote();
  const createFolder = useCreateFolder();
  const deleteFolder = useDeleteFolder();
  const deleteNote = useDeleteNote();

  const toggleFolder = (folderId: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) next.delete(folderId);
      else next.add(folderId);
      return next;
    });
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return;
    await createFolder.mutateAsync({
      name: newFolderName.trim(),
      artist_id: scope.artistId,
      project_id: scope.projectId,
    });
    setNewFolderName("");
    setShowNewFolder(false);
  };

  const handleCreateNote = async (folderId?: string) => {
    const note = await createNote.mutateAsync({
      artist_id: scope.artistId,
      project_id: scope.projectId,
      folder_id: folderId,
    });
    if (note?.id) onSelectNote(note.id);
  };

  const rootFolders = folders.filter((f) => !f.parent_folder_id);
  const unfolderedNotes = notes.filter((n) => !n.folder_id);

  const renderFolder = (folder: NoteFolder, depth: number = 0) => {
    const isExpanded = expandedFolders.has(folder.id);
    const children = folders.filter((f) => f.parent_folder_id === folder.id);
    const folderNotes = notes.filter((n) => n.folder_id === folder.id);

    return (
      <div key={folder.id}>
        <div
          className="flex items-center gap-1 py-1 px-2 rounded hover:bg-muted/50 cursor-pointer group"
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() => toggleFolder(folder.id)}
        >
          {isExpanded ? <ChevronDown className="w-3 h-3 shrink-0" /> : <ChevronRight className="w-3 h-3 shrink-0" />}
          <Folder className="w-4 h-4 text-muted-foreground shrink-0" />
          <span className="text-sm truncate flex-1">{folder.name}</span>
          <div className="hidden group-hover:flex items-center gap-0.5">
            <Button size="icon" variant="ghost" className="h-5 w-5" onClick={(e) => { e.stopPropagation(); handleCreateNote(folder.id); }}>
              <FilePlus className="w-3 h-3" />
            </Button>
            <Button size="icon" variant="ghost" className="h-5 w-5 text-destructive" onClick={(e) => { e.stopPropagation(); deleteFolder.mutate(folder.id); }}>
              <Trash2 className="w-3 h-3" />
            </Button>
          </div>
        </div>
        {isExpanded && (
          <>
            {children.map((c) => renderFolder(c, depth + 1))}
            {folderNotes.map((n) => renderNote(n, depth + 1))}
          </>
        )}
      </div>
    );
  };

  const renderNote = (note: Note, depth: number = 0) => (
    <div
      key={note.id}
      className={cn(
        "flex items-center gap-1 py-1 px-2 rounded cursor-pointer group",
        selectedNoteId === note.id ? "bg-primary/10 text-primary" : "hover:bg-muted/50"
      )}
      style={{ paddingLeft: `${depth * 16 + 8}px` }}
      onClick={() => onSelectNote(note.id)}
    >
      <FileText className="w-4 h-4 text-muted-foreground shrink-0" />
      {note.pinned && <Pin className="w-3 h-3 text-amber-500 shrink-0" />}
      <span className="text-sm truncate flex-1">{note.title}</span>
      <Button size="icon" variant="ghost" className="h-5 w-5 hidden group-hover:flex text-destructive shrink-0"
        onClick={(e) => { e.stopPropagation(); deleteNote.mutate(note.id); }}>
        <Trash2 className="w-3 h-3" />
      </Button>
    </div>
  );

  return (
    <div className="w-64 border-r bg-muted/30 flex flex-col h-full">
      <div className="p-3 border-b flex items-center justify-between">
        <span className="text-sm font-medium">Notes</span>
        <div className="flex items-center gap-1">
          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => setShowNewFolder(true)}>
            <FolderPlus className="w-4 h-4" />
          </Button>
          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => handleCreateNote()}>
            <FilePlus className="w-4 h-4" />
          </Button>
        </div>
      </div>
      {showNewFolder && (
        <div className="p-2 border-b flex gap-1">
          <Input value={newFolderName} onChange={(e) => setNewFolderName(e.target.value)}
            placeholder="Folder name" className="h-7 text-xs"
            onKeyDown={(e) => { if (e.key === "Enter") handleCreateFolder(); if (e.key === "Escape") setShowNewFolder(false); }}
            autoFocus />
        </div>
      )}
      <div className="flex-1 overflow-y-auto p-1">
        {rootFolders.map((f) => renderFolder(f))}
        {unfolderedNotes.map((n) => renderNote(n))}
        {folders.length === 0 && notes.length === 0 && (
          <div className="text-center py-8 text-xs text-muted-foreground">
            <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No notes yet</p>
            <p>Click + to create one</p>
          </div>
        )}
      </div>
    </div>
  );
}
