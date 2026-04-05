import { Component, type ReactNode, useEffect, useRef } from "react";
import { useCreateBlockNote } from "@blocknote/react";
import { BlockNoteView } from "@blocknote/shadcn";
import "@blocknote/shadcn/style.css";

// Error boundary — BlockNote errors won't crash the whole page
class EditorErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-4 border rounded-lg bg-destructive/10 text-destructive text-sm">
          Editor failed to load. Try refreshing the page.
        </div>
      );
    }
    return this.props.children;
  }
}

interface Props {
  initialContent?: unknown[];
  onChange?: (content: unknown[]) => void;
  editable?: boolean;
}

function NotesEditorInner({ initialContent, onChange, editable = true }: Props) {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const editor = useCreateBlockNote({
    initialContent: initialContent && initialContent.length > 0
      ? (initialContent as Parameters<typeof useCreateBlockNote>[0]["initialContent"])
      : undefined,
  });

  useEffect(() => {
    if (!onChange) return;
    const handler = () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onChange(editor.document as unknown as unknown[]);
      }, 1000);
    };
    editor.onEditorContentChange(handler);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [editor, onChange]);

  // Respect system/app dark mode preference
  const prefersDark = typeof window !== "undefined" &&
    document.documentElement.classList.contains("dark");

  return (
    <div className="min-h-[300px] border rounded-lg overflow-hidden bg-background">
      <BlockNoteView editor={editor} editable={editable} theme={prefersDark ? "dark" : "light"} />
    </div>
  );
}

export default function NotesEditor(props: Props) {
  return (
    <EditorErrorBoundary>
      <NotesEditorInner {...props} />
    </EditorErrorBoundary>
  );
}
