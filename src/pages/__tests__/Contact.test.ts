import { describe, expect, it } from "vitest";

import { filterAttachments } from "../Contact";

const MB = 1024 * 1024;

const file = (name: string, size = 1024, type = "") => ({ name, size, type });

describe("filterAttachments", () => {
  it("accepts allowed extensions", () => {
    const incoming = [file("shot.png"), file("deal.pdf"), file("notes.docx")];
    const { accepted, error } = filterAttachments(0, incoming);
    expect(accepted).toHaveLength(3);
    expect(error).toBe("");
  });

  it("accepts an image by mime type even with an unfamiliar extension", () => {
    const { accepted, error } = filterAttachments(0, [file("scan.HEIF", 1024, "image/heif")]);
    expect(accepted).toHaveLength(1);
    expect(error).toBe("");
  });

  it("rejects disallowed types and reports why", () => {
    const { accepted, error } = filterAttachments(0, [file("payload.exe")]);
    expect(accepted).toEqual([]);
    expect(error).toBe("Only images and Word/PDF files are supported.");
  });

  it("rejects files over 2.5 MB, naming the file", () => {
    const { accepted, error } = filterAttachments(0, [file("big.pdf", 2.5 * MB + 1)]);
    expect(accepted).toEqual([]);
    expect(error).toBe("big.pdf is over the 2.5 MB limit.");
  });

  it("keeps the good files from a mixed batch", () => {
    const { accepted, error } = filterAttachments(0, [file("ok.png"), file("bad.exe")]);
    expect(accepted.map((f) => f.name)).toEqual(["ok.png"]);
    expect(error).toBe("Only images and Word/PDF files are supported.");
  });

  it("caps the batch at 3 files and says so", () => {
    const incoming = [file("a.png"), file("b.png"), file("c.png"), file("d.png")];
    const { accepted, error } = filterAttachments(0, incoming);
    expect(accepted.map((f) => f.name)).toEqual(["a.png", "b.png", "c.png"]);
    expect(error).toBe("You can attach up to 3 files.");
  });

  it("counts files already attached against the cap", () => {
    const { accepted, error } = filterAttachments(2, [file("c.png"), file("d.png")]);
    expect(accepted.map((f) => f.name)).toEqual(["c.png"]);
    expect(error).toBe("You can attach up to 3 files.");
  });

  it("accepts nothing once the cap is already reached", () => {
    const { accepted, error } = filterAttachments(3, [file("d.png")]);
    expect(accepted).toEqual([]);
    expect(error).toBe("You can attach up to 3 files.");
  });

  it("is a no-op for an empty batch", () => {
    const { accepted, error } = filterAttachments(0, []);
    expect(accepted).toEqual([]);
    expect(error).toBe("");
  });
});
