import { FileText, ListMusic } from "lucide-react";

interface SongMismatchComparisonProps {
  contractWorks: string[];
  statementSongs: string[];
  statementTotalCount: number;
}

const SongMismatchComparison = ({
  contractWorks,
  statementSongs,
  statementTotalCount,
}: SongMismatchComparisonProps) => {
  const previewCount = statementSongs.length;
  const moreCount = Math.max(0, statementTotalCount - previewCount);

  return (
    <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 text-foreground">
      <div className="rounded-md border border-border bg-secondary/50 p-3">
        <div className="mb-2 flex items-center gap-2 text-sm font-medium">
          <FileText className="h-4 w-4" />
          <span>From your contract</span>
        </div>
        {contractWorks.length === 0 ? (
          <p className="text-sm text-muted-foreground">No songs found.</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {contractWorks.map((title) => (
              <li key={title} className="truncate">
                {title}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="rounded-md border border-border bg-secondary/50 p-3">
        <div className="mb-2 flex items-center gap-2 text-sm font-medium">
          <ListMusic className="h-4 w-4" />
          <span>From your statement</span>
        </div>
        {statementSongs.length === 0 ? (
          <p className="text-sm text-muted-foreground">No songs found.</p>
        ) : (
          <>
            <ul className="space-y-1 text-sm">
              {statementSongs.map((title) => (
                <li key={title} className="truncate">
                  {title}
                </li>
              ))}
            </ul>
            {moreCount > 0 && (
              <p className="mt-2 text-xs text-muted-foreground">
                +{moreCount} more song{moreCount === 1 ? "" : "s"} not shown
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default SongMismatchComparison;
