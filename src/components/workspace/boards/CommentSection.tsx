import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Trash2 } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { useAuth } from "@/contexts/AuthContext";
import type { TaskComment } from "@/types/integrations";

interface CommentSectionProps {
  taskId: string;
  comments: TaskComment[];
  onAdd: (data: { taskId: string; content: string }) => void;
  onDelete: (commentId: string) => void;
}

export function CommentSection({
  taskId,
  comments,
  onAdd,
  onDelete,
}: CommentSectionProps) {
  const { user } = useAuth();
  const [content, setContent] = useState("");

  const handleSubmit = () => {
    if (!content.trim()) return;
    onAdd({ taskId, content: content.trim() });
    setContent("");
  };

  return (
    <div className="space-y-3">
      {/* Comment list */}
      {comments.length > 0 && (
        <div className="space-y-3 max-h-[300px] overflow-y-auto">
          {comments.map((comment) => (
            <div
              key={comment.id}
              className="bg-muted/50 rounded-lg p-3 space-y-1"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  {formatDistanceToNow(new Date(comment.created_at), {
                    addSuffix: true,
                  })}
                </span>
                {comment.user_id === user?.id && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 text-muted-foreground hover:text-destructive"
                    onClick={() => onDelete(comment.id)}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                )}
              </div>
              <p className="text-sm whitespace-pre-wrap">{comment.content}</p>
            </div>
          ))}
        </div>
      )}

      {/* Add comment */}
      <div className="space-y-2">
        <Textarea
          placeholder="Add a comment..."
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="min-h-[80px] text-sm"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              handleSubmit();
            }
          }}
        />
        <Button
          size="sm"
          onClick={handleSubmit}
          disabled={!content.trim()}
        >
          Add Comment
        </Button>
      </div>
    </div>
  );
}
