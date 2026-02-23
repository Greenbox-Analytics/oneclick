import { useState } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Upload, FileText, CheckCircle, XCircle, Loader2, FileSpreadsheet } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";

interface RoyaltyStatementUploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: string;
  onUploadComplete?: () => void;
}

interface UploadResult {
  filename: string;
  status: "success" | "error" | "uploading";
  error?: string;
}

export const RoyaltyStatementUploadModal = ({
  open,
  onOpenChange,
  projectId,
  onUploadComplete,
}: RoyaltyStatementUploadModalProps) => {
  const { toast } = useToast();
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);
  const [error, setError] = useState("");

  const normalizeFileName = (name: string) => name.trim().toLowerCase();

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    const files = Array.from(e.target.files);
    const validFiles = files.filter((file) => {
      const ext = file.name.toLowerCase();
      return ext.endsWith(".xlsx") || ext.endsWith(".csv");
    });
    const blockedNames = new Set<string>();
    const seenInBatch = new Set<string>();
    
    if (validFiles.length !== files.length) {
      setError("Only XLSX and CSV files are supported. Non-supported files were filtered out.");
    } else {
      setError("");
    }

    let projectFileNames = new Set<string>();

    try {
      const { data, error: projectFilesError } = await supabase
        .from('project_files')
        .select('file_name')
        .eq('project_id', projectId);

      if (projectFilesError) {
        throw projectFilesError;
      }

      projectFileNames = new Set((data || []).map(file => normalizeFileName(file.file_name)));
    } catch (projectCheckError) {
      console.error("Error checking duplicate royalty files:", projectCheckError);
    }

    const filteredFiles: File[] = [];
    for (const file of validFiles) {
      const normalized = normalizeFileName(file.name);
      if (seenInBatch.has(normalized) || projectFileNames.has(normalized)) {
        blockedNames.add(file.name);
        continue;
      }

      seenInBatch.add(normalized);
      filteredFiles.push(file);
    }

    if (blockedNames.size > 0) {
      toast({
        title: "Duplicate file names blocked",
        description: Array.from(blockedNames).join(", "),
        variant: "destructive",
        duration: Number.POSITIVE_INFINITY,
      });
    }
    
    setSelectedFiles(filteredFiles);
    setUploadResults([]);
    e.target.value = "";
  };

  const handleUpload = async () => {
    if (!selectedFiles.length) {
      setError("Please select at least one XLSX or CSV file");
      return;
    }

    setUploading(true);
    setError("");
    
    // Initialize upload results
    const initialResults: UploadResult[] = selectedFiles.map((file) => ({
      filename: file.name,
      status: "uploading",
    }));
    setUploadResults(initialResults);

    // Upload files one by one
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];

      try {
        // Upload to Supabase storage
        const filePath = `${projectId}/royalty_statement/${Date.now()}_${file.name}`;
        const { error: uploadError } = await supabase.storage
          .from('project-files')
          .upload(filePath, file);

        if (uploadError) throw uploadError;

        // Get public URL
        const { data: urlData } = supabase.storage
          .from('project-files')
          .getPublicUrl(filePath);

        // Create database record
        const { error: dbError } = await supabase
          .from('project_files')
          .insert({
            project_id: projectId,
            file_name: file.name,
            file_url: urlData.publicUrl,
            file_path: filePath,
            folder_category: 'royalty_statement',
            file_size: file.size,
            file_type: file.type,
          });

        if (dbError) throw dbError;

        // Update result for this file
        setUploadResults((prev) =>
          prev.map((r, idx) =>
            idx === i
              ? {
                  filename: file.name,
                  status: "success",
                }
              : r
          )
        );
      } catch (err) {
        console.error(`Error uploading ${file.name}:`, err);

        const rawErrorMessage = err instanceof Error ? err.message : "Upload failed";
        const friendlyErrorMessage = rawErrorMessage.toLowerCase().includes("duplicate") || rawErrorMessage.toLowerCase().includes("unique")
          ? `A file named "${file.name}" already exists in this project.`
          : rawErrorMessage;
        
        // Update result for this file
        setUploadResults((prev) =>
          prev.map((r, idx) =>
            idx === i
              ? {
                  filename: file.name,
                  status: "error",
                  error: friendlyErrorMessage,
                }
              : r
          )
        );
      }
    }

    setUploading(false);
    
    // Call completion callback when uploads finish
    onUploadComplete?.();
    
    // Show success toast
    const successCount = uploadResults.filter((r) => r.status === "success").length;
    if (successCount > 0) {
      toast({
        title: "Success",
        description: `${successCount} file${successCount > 1 ? 's' : ''} uploaded successfully`,
      });
    }
  };

  const handleClose = () => {
    if (!uploading) {
      setSelectedFiles([]);
      setUploadResults([]);
      setError("");
      onOpenChange(false);
    }
  };

  const successCount = uploadResults.filter((r) => r.status === "success").length;
  const errorCount = uploadResults.filter((r) => r.status === "error").length;

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Royalty Statements
          </DialogTitle>
          <DialogDescription>
            Upload one or multiple XLSX or CSV royalty statement files.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4 overflow-hidden">
          {/* File Input */}
          <div className="space-y-2">
            <Label htmlFor="royalty-files">Select XLSX or CSV Files</Label>
            <div className="flex items-center justify-center gap-2 mb-3">
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-md">
                <FileSpreadsheet className="w-3.5 h-3.5 text-green-600 dark:text-green-400" />
                <span className="text-xs font-medium text-green-700 dark:text-green-300">XLSX or CSV only</span>
              </div>
            </div>
            <Input
              id="royalty-files"
              type="file"
              accept=".xlsx,.csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv"
              multiple
              onChange={handleFileSelect}
              disabled={uploading}
              className="cursor-pointer"
            />
            {selectedFiles.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">
                  {selectedFiles.length} file{selectedFiles.length > 1 ? "s" : ""} selected
                </p>
                <div className="max-h-32 overflow-y-auto overflow-x-hidden rounded-md border bg-muted/30 p-2">
                  <ul className="space-y-1 text-sm min-w-0">
                    {selectedFiles.map((file, idx) => (
                      <li key={`${file.name}-${idx}`} className="truncate text-foreground" title={file.name}>
                        {file.name}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* Error Alert */}
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Upload Progress */}
          {uploadResults.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">Upload Progress</span>
                <span className="text-muted-foreground">
                  {successCount + errorCount} / {uploadResults.length} completed
                </span>
              </div>

              <Progress
                value={((successCount + errorCount) / uploadResults.length) * 100}
                className="h-2"
              />

              {/* File Results */}
              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {uploadResults.map((result, idx) => (
                  <div
                    key={idx}
                    className="flex items-start gap-3 p-3 rounded-lg border bg-card"
                  >
                    {result.status === "uploading" && (
                      <Loader2 className="w-5 h-5 text-blue-500 animate-spin flex-shrink-0 mt-0.5" />
                    )}
                    {result.status === "success" && (
                      <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                    )}
                    {result.status === "error" && (
                      <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    )}

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <FileSpreadsheet className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                        <p className="text-sm font-medium truncate" title={result.filename}>{result.filename}</p>
                      </div>

                      {result.status === "uploading" && (
                        <p className="text-xs text-muted-foreground mt-1">
                          Uploading...
                        </p>
                      )}

                      {result.status === "success" && (
                        <p className="text-xs text-green-600 mt-1">
                          ✓ Uploaded successfully
                        </p>
                      )}

                      {result.status === "error" && result.error && (
                        <p className="text-xs text-red-600 mt-1">✗ {result.error}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Summary */}
              {!uploading && uploadResults.length > 0 && (
                <div className="flex items-center gap-4 text-sm pt-2 border-t">
                  {successCount > 0 && (
                    <span className="text-green-600 font-medium">
                      ✓ {successCount} succeeded
                    </span>
                  )}
                  {errorCount > 0 && (
                    <span className="text-red-600 font-medium">
                      ✗ {errorCount} failed
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <DialogFooter>
          {uploadResults.length > 0 && !uploading ? (
            <Button onClick={handleClose} className="w-full">
              <CheckCircle className="w-4 h-4 mr-2" />
              Done
            </Button>
          ) : (
            <>
              <Button variant="outline" onClick={handleClose} disabled={uploading}>
                Cancel
              </Button>
              <Button
                onClick={handleUpload}
                disabled={selectedFiles.length === 0 || uploading}
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload {selectedFiles.length > 0 && `(${selectedFiles.length})`}
                  </>
                )}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
