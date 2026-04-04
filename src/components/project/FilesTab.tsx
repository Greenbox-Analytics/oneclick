import { useState, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Loader2, ChevronRight, Upload, FileText, Search, Download, ExternalLink,
} from "lucide-react";
import { toast } from "sonner";

interface FilesTabProps {
  projectId: string;
  userRole: string | null;
}

const FOLDER_CATEGORIES = [
  { key: "contract", label: "Contracts" },
  { key: "split_sheet", label: "Split Sheets" },
  { key: "royalty_statement", label: "Royalty Statements" },
  { key: "other", label: "Other" },
];

const canEdit = (role: string | null) => role === "owner" || role === "admin" || role === "editor";

export default function FilesTab({ projectId, userRole }: FilesTabProps) {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [uploadingCategory, setUploadingCategory] = useState<string | null>(null);
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    contract: true,
    split_sheet: true,
    royalty_statement: true,
    other: true,
  });
  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  // Fetch project files
  const { data: files, isLoading } = useQuery({
    queryKey: ["project-files-tab", projectId],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("project_files")
        .select("*")
        .eq("project_id", projectId)
        .order("created_at", { ascending: false });
      if (error) throw error;
      return data || [];
    },
    enabled: !!projectId,
  });

  // Fetch work_files links to show relevant works
  const { data: workFileLinks } = useQuery({
    queryKey: ["work-file-links-for-project", projectId],
    queryFn: async () => {
      if (!files || files.length === 0) return [];
      const fileIds = files.map((f) => f.id);
      // Query work_files join with works_registry to get work titles
      const { data, error } = await (supabase as any)
        .from("work_files")
        .select("file_id, work_id, works_registry(id, title)")
        .in("file_id", fileIds);
      if (error) return [];
      return data || [];
    },
    enabled: !!files && files.length > 0,
  });

  // Build file -> works lookup
  const fileWorksMap = new Map<string, { workId: string; title: string }[]>();
  if (workFileLinks) {
    for (const link of workFileLinks) {
      const workInfo = link.works_registry;
      if (!workInfo) continue;
      if (!fileWorksMap.has(link.file_id)) fileWorksMap.set(link.file_id, []);
      fileWorksMap.get(link.file_id)!.push({ workId: workInfo.id, title: workInfo.title });
    }
  }

  const handleUpload = async (category: string, event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadingCategory(category);
    try {
      const filePath = `${projectId}/${category}/${Date.now()}_${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from("project-files")
        .upload(filePath, file);
      if (uploadError) throw uploadError;

      const { data: urlData } = supabase.storage.from("project-files").getPublicUrl(filePath);

      const { error: dbError } = await supabase
        .from("project_files")
        .insert({
          project_id: projectId,
          file_name: file.name,
          file_url: urlData.publicUrl,
          folder_category: category,
          file_size: file.size,
          file_type: file.type,
        });

      if (dbError) {
        await supabase.storage.from("project-files").remove([filePath]);
        throw dbError;
      }

      queryClient.invalidateQueries({ queryKey: ["project-files-tab", projectId] });
      toast.success("File uploaded");
    } catch (error: any) {
      toast.error(error.message || "Failed to upload file");
    } finally {
      setUploadingCategory(null);
      event.target.value = "";
    }
  };

  const handleDownload = async (file: any) => {
    try {
      // Use the file_url directly if available
      if (file.file_url) {
        window.open(file.file_url, "_blank");
        return;
      }
    } catch {
      toast.error("Failed to download file");
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const allFiles = files || [];
  const filteredFiles = search
    ? allFiles.filter((f) => f.file_name.toLowerCase().includes(search.toLowerCase()))
    : allFiles;

  const filesByCategory = new Map<string, typeof allFiles>();
  for (const cat of FOLDER_CATEGORIES) {
    filesByCategory.set(
      cat.key,
      filteredFiles.filter((f) => f.folder_category === cat.key)
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search files..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      {allFiles.length === 0 && !search ? (
        <div className="flex flex-col items-center justify-center py-16 gap-2">
          <FileText className="w-12 h-12 text-muted-foreground/40" />
          <p className="text-muted-foreground">No files yet. Upload your first file.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {FOLDER_CATEGORIES.map((cat) => {
            const catFiles = filesByCategory.get(cat.key) || [];
            return (
              <Collapsible
                key={cat.key}
                open={openSections[cat.key]}
                onOpenChange={(open) =>
                  setOpenSections((prev) => ({ ...prev, [cat.key]: open }))
                }
              >
                <Card className="overflow-hidden">
                  <CollapsibleTrigger className="flex items-center justify-between w-full p-3 hover:bg-muted/50 transition-colors">
                    <div className="flex items-center gap-2">
                      <ChevronRight
                        className={`w-4 h-4 text-muted-foreground transition-transform ${
                          openSections[cat.key] ? "rotate-90" : ""
                        }`}
                      />
                      <span className="text-sm font-medium">{cat.label}</span>
                      <span className="text-xs text-muted-foreground">({catFiles.length})</span>
                    </div>
                    {canEdit(userRole) && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 text-xs"
                        disabled={uploadingCategory === cat.key}
                        onClick={(e) => {
                          e.stopPropagation();
                          fileInputRefs.current[cat.key]?.click();
                        }}
                      >
                        {uploadingCategory === cat.key ? (
                          <Loader2 className="w-3 h-3 animate-spin mr-1" />
                        ) : (
                          <Upload className="w-3 h-3 mr-1" />
                        )}
                        Upload
                      </Button>
                    )}
                    <input
                      ref={(el) => { fileInputRefs.current[cat.key] = el; }}
                      type="file"
                      className="hidden"
                      onChange={(e) => handleUpload(cat.key, e)}
                    />
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    {catFiles.length === 0 ? (
                      <p className="px-3 pb-3 text-xs text-muted-foreground">No files in this folder.</p>
                    ) : (
                      <div className="border-t border-border">
                        {catFiles.map((file) => {
                          const linkedWorks = fileWorksMap.get(file.id);
                          return (
                            <div
                              key={file.id}
                              className="flex items-center justify-between px-3 py-2 hover:bg-muted/30 border-b border-border last:border-b-0"
                            >
                              <div className="flex-1 min-w-0">
                                <p className="text-sm text-foreground truncate">{file.file_name}</p>
                                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                  <span>{new Date(file.created_at).toLocaleDateString()}</span>
                                  {file.file_size && (
                                    <span>{(file.file_size / 1024).toFixed(0)} KB</span>
                                  )}
                                  {linkedWorks && linkedWorks.length > 0 && (
                                    <span>
                                      Works:{" "}
                                      {linkedWorks.map((w, i) => (
                                        <span key={w.workId}>
                                          {i > 0 && ", "}
                                          <span className="text-primary hover:underline cursor-pointer">
                                            {w.title}
                                          </span>
                                        </span>
                                      ))}
                                    </span>
                                  )}
                                </div>
                              </div>
                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 w-7 p-0 shrink-0"
                                onClick={() => handleDownload(file)}
                              >
                                {file.file_url ? (
                                  <ExternalLink className="w-3.5 h-3.5" />
                                ) : (
                                  <Download className="w-3.5 h-3.5" />
                                )}
                              </Button>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </CollapsibleContent>
                </Card>
              </Collapsible>
            );
          })}
        </div>
      )}
    </div>
  );
}
