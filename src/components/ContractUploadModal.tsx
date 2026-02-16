import { useState } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

interface ContractUploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: string;
  onUploadComplete?: () => void;
}

interface UploadResult {
  filename: string;
  status: "success" | "error" | "uploading";
  contract_id?: string;
  total_chunks?: number;
  error?: string;
}

export const ContractUploadModal = ({
  open,
  onOpenChange,
  projectId,
  onUploadComplete,
}: ContractUploadModalProps) => {
  const { user } = useAuth();
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);
  const [error, setError] = useState("");

  const normalizeFileName = (name: string) => name.trim().toLowerCase();

  const showPersistentDuplicateToast = (message: string) => {
    toast.error(message, {
      duration: Infinity,
      closeButton: true,
      style: {
        background: "hsl(var(--destructive))",
        color: "hsl(var(--destructive-foreground))",
        border: "1px solid hsl(var(--destructive))",
      },
    });
  };

  const getProjectFileNames = async (): Promise<Set<string>> => {
    const response = await fetch(`${API_URL}/files/${projectId}`);
    if (!response.ok) {
      throw new Error("Failed to check existing project files");
    }

    const files = await response.json();
    return new Set((files || []).map((f: { file_name: string }) => normalizeFileName(f.file_name)));
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    const files = Array.from(e.target.files);
    const pdfFiles = files.filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    const blockedNames = new Set<string>();
    const seenInBatch = new Set<string>();

    if (pdfFiles.length !== files.length) {
      setError("Only PDF files are supported. Non-PDF files were filtered out.");
    } else {
      setError("");
    }

    let projectFileNames = new Set<string>();

    try {
      projectFileNames = await getProjectFileNames();
    } catch (projectCheckError) {
      console.error("Error checking duplicate contract files:", projectCheckError);
    }

    const filteredFiles: File[] = [];
    for (const file of pdfFiles) {
      const normalized = normalizeFileName(file.name);
      if (seenInBatch.has(normalized) || projectFileNames.has(normalized)) {
        blockedNames.add(file.name);
        continue;
      }

      seenInBatch.add(normalized);
      filteredFiles.push(file);
    }

    if (blockedNames.size > 0) {
      showPersistentDuplicateToast(`Duplicate file name(s) blocked: ${Array.from(blockedNames).join(", ")}`);
    }

    setSelectedFiles(filteredFiles);
    setUploadResults([]);
    e.target.value = "";
  };

  const handleUpload = async () => {
    if (!selectedFiles.length || !user) {
      setError("Please select at least one PDF file");
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

    let localSuccessCount = 0;

    // Upload files one by one
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      const formData = new FormData();
      formData.append("file", file);
      formData.append("project_id", projectId);
      formData.append("user_id", user.id);

      try {
        const response = await fetch(`${API_URL}/contracts/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || "Upload failed");
        }

        const result = await response.json();
        localSuccessCount++;

        // Update result for this file
        setUploadResults((prev) =>
          prev.map((r, idx) =>
            idx === i
              ? {
                  filename: file.name,
                  status: "success",
                  contract_id: result.contract_id,
                  total_chunks: result.total_chunks,
                }
              : r
          )
        );
      } catch (err) {
        console.error(`Error uploading ${file.name}:`, err);
        
        // Update result for this file
        setUploadResults((prev) =>
          prev.map((r, idx) =>
            idx === i
              ? {
                  filename: file.name,
                  status: "error",
                  error: err instanceof Error ? err.message : "Upload failed",
                }
              : r
          )
        );
      }
    }

    setUploading(false);
    
    // Only close/notify if at least one file succeeded
    if (localSuccessCount > 0) {
      toast.success(`Successfully uploaded ${localSuccessCount} file${localSuccessCount !== 1 ? 's' : ''}`);
      onUploadComplete?.();
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
  const uploadingCount = uploadResults.filter((r) => r.status === "uploading").length;

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload Contracts
          </DialogTitle>
          <DialogDescription>
            Upload one or multiple PDF contract files. Each will be processed and indexed for AI search.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* File Input */}
          <div className="space-y-2">
            <Label htmlFor="contract-files">Select PDF Files</Label>
            <div className="flex items-center justify-center gap-2 mb-3">
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-md">
                <FileText className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                <span className="text-xs font-medium text-blue-700 dark:text-blue-300">PDF only</span>
              </div>
            </div>
            <Input
              id="contract-files"
              type="file"
              accept=".pdf"
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
                <div className="max-h-32 overflow-y-auto rounded-md border bg-muted/30 p-2">
                  <ul className="space-y-1 text-sm">
                    {selectedFiles.map((file, idx) => (
                      <li key={`${file.name}-${idx}`} className="truncate text-foreground">
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
                      <div className="flex items-start gap-2">
                        <FileText className="w-4 h-4 text-muted-foreground flex-shrink-0 mt-0.5" />
                        <p className="text-sm font-medium break-words">{result.filename}</p>
                      </div>

                      {result.status === "uploading" && (
                        <p className="text-xs text-muted-foreground mt-1">
                          Uploading...
                        </p>
                      )}

                      {result.status === "success" && result.total_chunks && (
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
