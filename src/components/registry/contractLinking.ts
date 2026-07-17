import { supabase } from "@/integrations/supabase/client";
import type { ParsedParty } from "@/hooks/useParseContractSplits";

// A contract queued for AI split parsing. Splits are often spread across
// several contracts (producer deal, feature deal, …) that only together
// account for 100% — the queue lets the user parse them all and merge.
export interface QueuedContract {
  id: string;
  kind: "upload" | "project";
  file?: File;
  contractFileId?: string;
  displayName: string;
  status: "pending" | "parsing" | "done" | "error";
  error?: string;
  parties?: ParsedParty[]; // raw parse result, kept for provenance + re-merge
  mainArtistFound?: boolean;
}

/**
 * Split the queued contracts into the two shapes the wizard needs to link them
 * to the new work's Related documents:
 *  - `existingFileIds` — contracts picked from the project already live in
 *    project_files, so they can be linked by id directly. Deduped so a repeated
 *    pick can't hit the work_files UNIQUE(work_id, file_id) constraint.
 *  - `uploads` — contracts uploaded fresh in the wizard exist only in memory;
 *    they must be persisted to project_files before a link is possible.
 * Contracts missing the field their kind needs (a project pick without an id, an
 * upload without a file) are dropped — there's nothing linkable there.
 */
export function partitionContractsForLinking(queued: QueuedContract[]): {
  existingFileIds: string[];
  uploads: Array<{ file: File; displayName: string }>;
} {
  const existingFileIds: string[] = [];
  const seen = new Set<string>();
  const uploads: Array<{ file: File; displayName: string }> = [];
  for (const qc of queued) {
    if (qc.kind === "project") {
      if (qc.contractFileId && !seen.has(qc.contractFileId)) {
        seen.add(qc.contractFileId);
        existingFileIds.push(qc.contractFileId);
      }
    } else if (qc.file) {
      uploads.push({ file: qc.file, displayName: qc.displayName });
    }
  }
  return { existingFileIds, uploads };
}

/**
 * Save an uploaded contract PDF into the project's Files as a contract and return
 * its project_files id. Mirrors WorkEditor's upload-then-link pipeline: SHA-256
 * dedup (re-use the existing row if the exact file is already in the project),
 * storage upload, then the project_files insert. On insert failure the just-
 * uploaded object is removed so we don't leave orphaned storage bytes.
 */
export async function persistContractToProject(file: File, projectId: string): Promise<string> {
  const hashBuffer = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  const contentHash = Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  const { data: existing } = await supabase
    .from("project_files")
    .select("id")
    .eq("project_id", projectId)
    .eq("content_hash", contentHash)
    .limit(1);
  if (existing && existing.length > 0) return existing[0].id;

  const filePath = `${projectId}/contract/${Date.now()}_${file.name}`;
  const { error: uploadError } = await supabase.storage
    .from("project-files")
    .upload(filePath, file);
  if (uploadError) throw uploadError;
  const { data: urlData } = supabase.storage.from("project-files").getPublicUrl(filePath);
  const { data: inserted, error: dbError } = await supabase
    .from("project_files")
    .insert({
      project_id: projectId,
      file_name: file.name,
      file_url: urlData.publicUrl,
      file_path: filePath,
      folder_category: "contract",
      file_size: file.size,
      file_type: file.type,
      content_hash: contentHash,
    })
    .select("id")
    .single();
  if (dbError) {
    await supabase.storage.from("project-files").remove([filePath]);
    throw dbError;
  }
  return inserted.id;
}
