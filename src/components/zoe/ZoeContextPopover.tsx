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

function CheckMark() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
      <path d="M5 13l4 4L19 7" />
    </svg>
  );
}

export function ZoeContextPopover({
  children,
  contextTree,
  checkedArtistIds,
  setCheckedArtistIds,
  checkedProjectIds,
  setCheckedProjectIds,
  projectDocuments,
  selectedContracts,
  onSelectedContractsChange,
  uploadModalOpen,
  onUploadModalOpenChange,
  onUploadComplete,
}: ZoeContextPopoverProps) {
  const [open, setOpen] = useState(false);
  const [artistQuery, setArtistQuery] = useState("");
  const [projectQuery, setProjectQuery] = useState("");
  const [docQuery, setDocQuery] = useState("");

  // ARTISTS — filtered by the artist search
  const aq = artistQuery.trim().toLowerCase();
  const artists = aq
    ? contextTree.artists.filter((a) => a.name.toLowerCase().includes(aq))
    : contextTree.artists;

  // PROJECTS — only for checked artists, filtered by the project search
  const pq = projectQuery.trim().toLowerCase();
  const projects = useMemo(
    () =>
      contextTree.projects.filter(
        (p) => checkedArtistIds.includes(p.artist_id) && (!pq || p.name.toLowerCase().includes(pq))
      ),
    [contextTree.projects, checkedArtistIds, pq]
  );

  // DOCUMENTS — for checked projects, filtered by the document search
  const dq = docQuery.trim().toLowerCase();
  const docs = useMemo(
    () =>
      checkedProjectIds
        .flatMap((pid) => projectDocuments[pid] || [])
        .filter((d) => !dq || d.file_name.toLowerCase().includes(dq)),
    [checkedProjectIds, projectDocuments, dq]
  );

  const selectedDocCount = docs.filter((d) => selectedContracts.includes(d.id)).length;
  const uploadProjectId = checkedProjectIds[0];

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>{children}</PopoverTrigger>
        <PopoverContent align="end" sideOffset={8} className="zoe-cmp w-[360px] p-0" style={{ zIndex: 50 }}>
          <div className="zoe-cmp-scroll">
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
              <div className="zoe-cmp-search">
                <Search />
                <input
                  value={artistQuery}
                  onChange={(e) => setArtistQuery(e.target.value)}
                  placeholder="Search artists…"
                />
              </div>
              <div className="zoe-cmp-list">
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
                          {checked && <CheckMark />}
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
                <div className="zoe-cmp-search">
                  <Search />
                  <input
                    value={projectQuery}
                    onChange={(e) => setProjectQuery(e.target.value)}
                    placeholder="Search projects…"
                  />
                </div>
                <div className="zoe-cmp-list">
                  {projects.length === 0 ? (
                    <p className="zoe-cmp-empty">No projects for the selected artists</p>
                  ) : (
                    projects.map((p) => {
                      const checked = checkedProjectIds.includes(p.id);
                      return (
                        <label key={p.id} className={`zoe-cmp-row ${checked ? "is-on" : ""}`}>
                          <span className={`zoe-cmp-check ${checked ? "is-on" : ""}`}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => setCheckedProjectIds((prev) => toggle(prev, p.id))}
                            />
                            {checked && <CheckMark />}
                          </span>
                          <span className="zoe-cmp-rowtext">
                            <span className="zoe-cmp-name">{p.name}</span>
                            <span className="zoe-cmp-meta">
                              {p.doc_count} doc{p.doc_count === 1 ? "" : "s"}
                            </span>
                          </span>
                        </label>
                      );
                    })
                  )}
                </div>
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
                <div className="zoe-cmp-search">
                  <Search />
                  <input
                    value={docQuery}
                    onChange={(e) => setDocQuery(e.target.value)}
                    placeholder="Search documents…"
                  />
                </div>
                <div className="zoe-cmp-list">
                  {docs.length === 0 ? (
                    <p className="zoe-cmp-empty">No documents in the selected projects</p>
                  ) : (
                    docs.map((d) => {
                      const checked = selectedContracts.includes(d.id);
                      return (
                        <label key={d.id} className={`zoe-cmp-row ${checked ? "is-on" : ""}`}>
                          <span className={`zoe-cmp-check ${checked ? "is-on" : ""}`}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => onSelectedContractsChange(toggle(selectedContracts, d.id))}
                            />
                            {checked && <CheckMark />}
                          </span>
                          <span className="zoe-cmp-rowtext">
                            <span className="zoe-cmp-name" title={d.file_name}>
                              {d.file_name}
                            </span>
                            {d.page_count != null && (
                              <span className="zoe-cmp-meta">
                                {d.page_count} page{d.page_count === 1 ? "" : "s"}
                              </span>
                            )}
                          </span>
                          <span className="zoe-cmp-badge">
                            {d.folder_category === "split_sheet" ? "SPLIT" : "CONTRACT"}
                          </span>
                        </label>
                      );
                    })
                  )}
                </div>
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
