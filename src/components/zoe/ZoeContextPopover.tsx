import { useMemo, useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Search } from "lucide-react";
import { ContractUploadModal } from "@/components/ContractUploadModal";

interface CtxArtist {
  id: string;
  name: string;
  project_count: number;
}
interface CtxProject {
  id: string;
  name: string;
  artist_id: string;
  doc_count: number;
}
interface CtxDoc {
  id: string;
  file_name: string;
  project_id: string;
  folder_category?: string;
  page_count?: number | null;
}

interface ZoeContextPopoverProps {
  children: React.ReactNode;
  contextTree: { artists: CtxArtist[]; projects: CtxProject[] };
  checkedArtistIds: string[];
  setCheckedArtistIds: (updater: string[] | ((prev: string[]) => string[])) => void;
  checkedProjectIds: string[];
  setCheckedProjectIds: (updater: string[] | ((prev: string[]) => string[])) => void;
  projectDocuments: Record<string, CtxDoc[]>;
  knownContracts: { id: string; file_name: string }[];
  selectedContracts: string[];
  onSelectedContractsChange: (contracts: string[]) => void;
  uploadModalOpen: boolean;
  onUploadModalOpenChange: (open: boolean) => void;
  onUploadComplete: () => void;
}

function initials(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() || "")
    .join("");
}

function toggle(list: string[], id: string): string[] {
  return list.includes(id) ? list.filter((x) => x !== id) : [...list, id];
}

export function ZoeContextPopover({
  children,
  contextTree,
  checkedArtistIds,
  setCheckedArtistIds,
  checkedProjectIds,
  setCheckedProjectIds,
  projectDocuments,
  knownContracts,
  selectedContracts,
  onSelectedContractsChange,
  uploadModalOpen,
  onUploadModalOpenChange,
  onUploadComplete,
}: ZoeContextPopoverProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const q = query.trim().toLowerCase();
  const artistNameById = useMemo(
    () => Object.fromEntries(contextTree.artists.map((a) => [a.id, a.name])),
    [contextTree.artists]
  );
  const projectById = useMemo(
    () => Object.fromEntries(contextTree.projects.map((p) => [p.id, p])),
    [contextTree.projects]
  );

  // ARTISTS — filtered by search
  const artists = q
    ? contextTree.artists.filter((a) => a.name.toLowerCase().includes(q))
    : contextTree.artists;

  // PROJECTS — only for checked artists, filtered by search, grouped by artist
  const projectGroups = useMemo(() => {
    const visible = contextTree.projects.filter(
      (p) => checkedArtistIds.includes(p.artist_id) && (!q || p.name.toLowerCase().includes(q))
    );
    const byArtist: Record<string, CtxProject[]> = {};
    for (const p of visible) (byArtist[p.artist_id] ||= []).push(p);
    return checkedArtistIds
      .filter((aid) => byArtist[aid]?.length)
      .map((aid) => ({ artistId: aid, artistName: artistNameById[aid] || "Artist", projects: byArtist[aid] }));
  }, [contextTree.projects, checkedArtistIds, q, artistNameById]);

  // DOCUMENTS — for checked projects, grouped by artist — project
  const docGroups = useMemo(() => {
    return checkedProjectIds
      .map((pid) => {
        const proj = projectById[pid];
        const docs = projectDocuments[pid] || [];
        return proj && docs.length
          ? {
              projectId: pid,
              label: `${(artistNameById[proj.artist_id] || "Artist").toUpperCase()} — ${proj.name.toUpperCase()}`,
              docs,
            }
          : null;
      })
      .filter(Boolean) as { projectId: string; label: string; docs: CtxDoc[] }[];
  }, [checkedProjectIds, projectById, projectDocuments, artistNameById]);

  const selectedDocCount = docGroups.reduce(
    (n, g) => n + g.docs.filter((d) => selectedContracts.includes(d.id)).length,
    0
  );
  const uploadProjectId = checkedProjectIds[0];

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>{children}</PopoverTrigger>
        <PopoverContent align="end" sideOffset={8} className="zoe-cmp w-[360px] p-0" style={{ zIndex: 50 }}>
          <div className="zoe-cmp-head">
            <div className="zoe-cmp-title">Comparison context</div>
            <p className="zoe-cmp-sub">Check multiple artists &amp; projects to analyze their contracts side by side.</p>
            <div className="zoe-cmp-search">
              <Search />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search artists or projects…"
              />
            </div>
          </div>

          <div className="zoe-cmp-scroll">
            {/* SELECTED — always shows the current comparison set, regardless of which filters are open */}
            {selectedContracts.length > 0 && (
              <div className="zoe-cmp-section">
                <div className="zoe-cmp-shead">
                  <span>
                    Selected · <b>{selectedContracts.length}</b> contract{selectedContracts.length === 1 ? "" : "s"}
                  </span>
                  <button className="zoe-cmp-clear" onClick={() => onSelectedContractsChange([])}>
                    Deselect all
                  </button>
                </div>
                <div className="zoe-cmp-selchips">
                  {selectedContracts.map((id) => {
                    const label = knownContracts.find((c) => c.id === id)?.file_name || id;
                    return (
                      <span key={id} className="zoe-cmp-selchip" title={label}>
                        <span className="zoe-cmp-selchip-name">{label}</span>
                        <button
                          onClick={() => onSelectedContractsChange(selectedContracts.filter((x) => x !== id))}
                          title="Remove from selection"
                        >
                          ✕
                        </button>
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {/* ARTISTS */}
            <div className="zoe-cmp-section">
              <div className="zoe-cmp-shead">
                <span>
                  Artists · <b>{checkedArtistIds.length}</b> selected
                </span>
                {checkedArtistIds.length > 0 && (
                  <button className="zoe-cmp-clear" onClick={() => setCheckedArtistIds([])}>
                    Clear
                  </button>
                )}
              </div>
              {artists.length === 0 ? (
                <p className="zoe-cmp-empty">No artists found</p>
              ) : (
                artists.map((a) => {
                  const checked = checkedArtistIds.includes(a.id);
                  return (
                    <label key={a.id} className={`zoe-cmp-row ${checked ? "is-on" : ""}`}>
                      <span className={`zoe-cmp-check ${checked ? "is-on" : ""}`}>
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => setCheckedArtistIds((prev) => toggle(prev, a.id))}
                        />
                        {checked && (
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                            <path d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </span>
                      <span className="zoe-cmp-avatar">{initials(a.name)}</span>
                      <span className="zoe-cmp-rowtext">
                        <span className="zoe-cmp-name">{a.name}</span>
                        <span className="zoe-cmp-meta">
                          {a.project_count} project{a.project_count === 1 ? "" : "s"}
                        </span>
                      </span>
                    </label>
                  );
                })
              )}
            </div>

            {/* PROJECTS */}
            {checkedArtistIds.length > 0 && (
              <div className="zoe-cmp-section">
                <div className="zoe-cmp-shead">
                  <span>
                    Projects · <b>{checkedProjectIds.length}</b> selected
                  </span>
                  {checkedProjectIds.length > 0 && (
                    <button className="zoe-cmp-clear" onClick={() => setCheckedProjectIds([])}>
                      Clear
                    </button>
                  )}
                </div>
                {projectGroups.length === 0 ? (
                  <p className="zoe-cmp-empty">No projects for the selected artists</p>
                ) : (
                  projectGroups.map((g) => (
                    <div key={g.artistId}>
                      <div className="zoe-cmp-group">
                        <span className="zoe-cmp-dot" />
                        {g.artistName}
                      </div>
                      {g.projects.map((p) => {
                        const checked = checkedProjectIds.includes(p.id);
                        return (
                          <label key={p.id} className={`zoe-cmp-row ${checked ? "is-on" : ""}`}>
                            <span className={`zoe-cmp-check ${checked ? "is-on" : ""}`}>
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => setCheckedProjectIds((prev) => toggle(prev, p.id))}
                              />
                              {checked && (
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                  <path d="M5 13l4 4L19 7" />
                                </svg>
                              )}
                            </span>
                            <span className="zoe-cmp-rowtext">
                              <span className="zoe-cmp-name">{p.name}</span>
                              <span className="zoe-cmp-meta">
                                {p.doc_count} doc{p.doc_count === 1 ? "" : "s"}
                              </span>
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  ))
                )}
              </div>
            )}

            {/* DOCUMENTS */}
            {checkedProjectIds.length > 0 && (
              <div className="zoe-cmp-section">
                <div className="zoe-cmp-shead">
                  <span>
                    Documents · <b>{selectedDocCount}</b> selected
                  </span>
                  {uploadProjectId && (
                    <button className="zoe-cmp-clear" onClick={() => onUploadModalOpenChange(true)}>
                      + Upload
                    </button>
                  )}
                </div>
                {docGroups.length === 0 ? (
                  <p className="zoe-cmp-empty">No documents in the selected projects</p>
                ) : (
                  docGroups.map((g) => (
                    <div key={g.projectId}>
                      <div className="zoe-cmp-group">
                        <span className="zoe-cmp-dot" />
                        {g.label}
                      </div>
                      {g.docs.map((d) => {
                        const checked = selectedContracts.includes(d.id);
                        return (
                          <label key={d.id} className={`zoe-cmp-row ${checked ? "is-on" : ""}`}>
                            <span className={`zoe-cmp-check ${checked ? "is-on" : ""}`}>
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => onSelectedContractsChange(toggle(selectedContracts, d.id))}
                              />
                              {checked && (
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                  <path d="M5 13l4 4L19 7" />
                                </svg>
                              )}
                            </span>
                            <span className="zoe-cmp-rowtext">
                              <span className="zoe-cmp-name" title={d.file_name}>
                                {d.file_name}
                              </span>
                              {d.page_count != null && <span className="zoe-cmp-meta">{d.page_count} pages</span>}
                            </span>
                            <span className="zoe-cmp-badge">
                              {(d.folder_category === "split_sheet" ? "split sheet" : "contract").toUpperCase()}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </PopoverContent>
      </Popover>

      {uploadProjectId && (
        <ContractUploadModal
          open={uploadModalOpen}
          onOpenChange={onUploadModalOpenChange}
          projectId={uploadProjectId}
          onUploadComplete={onUploadComplete}
        />
      )}
    </>
  );
}
