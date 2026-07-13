import { useMutation } from "@tanstack/react-query";
import { API_URL, getAuthHeaders, ApiError } from "@/lib/apiFetch";

export interface ParsedParty {
  name: string;
  role: string;
  /** Alternate names from the contract (p/k/a, a/k/a, d/b/a, stage names). */
  aliases?: string[];
  master_pct: number;
  publishing_pct: number;
  /** SoundExchange (US digital performance) % — tracked separately, never counted in master. */
  soundexchange_pct?: number;
  is_main_artist: boolean;
}

export interface ParseContractSplitsResponse {
  parties: ParsedParty[];
  main_artist_found: boolean;
}

interface ParseInput {
  /** Either a fresh upload OR a project-file id — never both. */
  file?: File;
  contractFileId?: string;
  mainArtistName: string;
}

/**
 * Upload (or pick) a contract PDF and get back per-party master/publishing
 * splits. Hits POST /registry/parse-contract-splits — multipart in both
 * branches so the backend can switch on `file` vs `contract_file_id`.
 */
export function useParseContractSplits() {
  return useMutation<ParseContractSplitsResponse, Error, ParseInput>({
    mutationFn: async ({ file, contractFileId, mainArtistName }) => {
      if (!!file === !!contractFileId) {
        throw new Error("Provide exactly one of file or contractFileId");
      }
      const headers = await getAuthHeaders();
      const form = new FormData();
      if (file) form.append("file", file);
      if (contractFileId) form.append("contract_file_id", contractFileId);
      form.append("main_artist_name", mainArtistName || "");
      const res = await fetch(`${API_URL}/registry/parse-contract-splits`, {
        method: "POST",
        headers, // multipart Content-Type is auto-set by fetch when body is FormData
        body: form,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new ApiError(body.detail || `Request failed: ${res.status}`, res.status);
      }
      return res.json();
    },
  });
}
