// Authenticated PDF download, shared by the OneClick payout/receipt dialogs
// and the usage report buttons.
import { getAuthHeaders } from "@/lib/apiFetch";

export async function downloadPdf(url: string, filename: string): Promise<void> {
  const headers = await getAuthHeaders();
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(objectUrl);
}
