import { describe, it, expect } from "vitest";
import {
  partitionContractsForLinking,
  type QueuedContract,
} from "@/components/registry/contractLinking";

// Minimal factory — only the fields the partition helper reads matter.
function qc(over: Partial<QueuedContract> & Pick<QueuedContract, "kind">): QueuedContract {
  return {
    id: crypto.randomUUID(),
    displayName: "contract.pdf",
    status: "done",
    ...over,
  } as QueuedContract;
}

const fakeFile = (name: string) => new File(["x"], name, { type: "application/pdf" });

describe("partitionContractsForLinking", () => {
  it("returns existing project file ids for kind=project contracts", () => {
    const result = partitionContractsForLinking([
      qc({ kind: "project", contractFileId: "file-a" }),
      qc({ kind: "project", contractFileId: "file-b" }),
    ]);
    expect(result.existingFileIds).toEqual(["file-a", "file-b"]);
    expect(result.uploads).toEqual([]);
  });

  it("returns upload files for kind=upload contracts", () => {
    const f = fakeFile("deal.pdf");
    const result = partitionContractsForLinking([
      qc({ kind: "upload", file: f, displayName: "deal.pdf" }),
    ]);
    expect(result.existingFileIds).toEqual([]);
    expect(result.uploads).toEqual([{ file: f, displayName: "deal.pdf" }]);
  });

  it("dedupes repeated project file ids", () => {
    const result = partitionContractsForLinking([
      qc({ kind: "project", contractFileId: "file-a" }),
      qc({ kind: "project", contractFileId: "file-a" }),
    ]);
    expect(result.existingFileIds).toEqual(["file-a"]);
  });

  it("ignores project contracts missing a file id and uploads missing a file", () => {
    const result = partitionContractsForLinking([
      qc({ kind: "project", contractFileId: undefined }),
      qc({ kind: "upload", file: undefined }),
    ]);
    expect(result.existingFileIds).toEqual([]);
    expect(result.uploads).toEqual([]);
  });

  it("partitions a mixed queue and preserves order", () => {
    const f = fakeFile("feature.pdf");
    const result = partitionContractsForLinking([
      qc({ kind: "project", contractFileId: "file-a" }),
      qc({ kind: "upload", file: f, displayName: "feature.pdf" }),
      qc({ kind: "project", contractFileId: "file-b" }),
    ]);
    expect(result.existingFileIds).toEqual(["file-a", "file-b"]);
    expect(result.uploads).toEqual([{ file: f, displayName: "feature.pdf" }]);
  });
});
