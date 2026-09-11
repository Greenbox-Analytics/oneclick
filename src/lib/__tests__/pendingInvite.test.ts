import { beforeEach, describe, expect, it } from "vitest";
import {
  clearPendingInvite,
  normalizeInviteEmail,
  orgInvitePath,
  orgInviteTokenFromPath,
  readPendingInvite,
  stashPendingInvite,
} from "../pendingInvite";

describe("orgInviteTokenFromPath", () => {
  it("extracts the token from an org invite path", () => {
    expect(orgInviteTokenFromPath("/orgs/invite/abc123")).toBe("abc123");
    expect(orgInviteTokenFromPath("/orgs/invite/abc123?x=1")).toBe("abc123");
  });

  it("returns null for anything else", () => {
    expect(orgInviteTokenFromPath("/dashboard")).toBeNull();
    expect(orgInviteTokenFromPath("/orgs/invite/")).toBeNull();
    expect(orgInviteTokenFromPath(null)).toBeNull();
  });

  it("round-trips through orgInvitePath", () => {
    expect(orgInviteTokenFromPath(orgInvitePath("tok"))).toBe("tok");
  });
});

describe("normalizeInviteEmail", () => {
  it("keeps a plausible address, trimmed and case-preserved", () => {
    expect(normalizeInviteEmail(" Ada@Example.com ")).toBe("Ada@Example.com");
    expect(normalizeInviteEmail("a+team@b.c")).toBe("a+team@b.c");
  });

  it("rejects anything that is not one address", () => {
    expect(normalizeInviteEmail(null)).toBeNull();
    expect(normalizeInviteEmail(undefined)).toBeNull();
    expect(normalizeInviteEmail(42)).toBeNull();
    expect(normalizeInviteEmail("")).toBeNull();
    expect(normalizeInviteEmail("no-at-sign")).toBeNull();
    expect(normalizeInviteEmail("@b.c")).toBeNull();
    expect(normalizeInviteEmail("a@")).toBeNull();
    expect(normalizeInviteEmail("a@b@c")).toBeNull();
    expect(normalizeInviteEmail("a b@c.d")).toBeNull();
    expect(normalizeInviteEmail("a\n@c.d")).toBeNull();
    expect(normalizeInviteEmail(`${"a".repeat(251)}@b.c`)).toBeNull(); // 255 chars
  });
});

describe("pending invite stash", () => {
  beforeEach(() => sessionStorage.clear());

  it("is empty by default", () => {
    expect(readPendingInvite()).toBeNull();
  });

  it("stores and reads an unaccepted invite", () => {
    stashPendingInvite({ token: "tok", accepted: false });
    expect(readPendingInvite()).toEqual({ token: "tok", accepted: false, orgName: null, kind: null, email: null });
  });

  it("stores the org name once accepted", () => {
    stashPendingInvite({ token: "tok", accepted: true, orgName: "Acme", kind: "self_serve" });
    expect(readPendingInvite()).toEqual({
      token: "tok",
      accepted: true,
      orgName: "Acme",
      kind: "self_serve",
      email: null,
    });
  });

  it("carries the invitee email from the emailed link", () => {
    stashPendingInvite({ token: "tok", accepted: false, email: "a+b@c.d" });
    expect(readPendingInvite()?.email).toBe("a+b@c.d");
  });

  it("drops a malformed stored email instead of prefilling it", () => {
    sessionStorage.setItem(
      "msanii_pending_org_invite",
      JSON.stringify({ token: "tok", accepted: false, email: "not an email" }),
    );
    expect(readPendingInvite()?.email).toBeNull();
  });

  it("clears", () => {
    stashPendingInvite({ token: "tok", accepted: false });
    clearPendingInvite();
    expect(readPendingInvite()).toBeNull();
  });

  it("ignores garbage", () => {
    sessionStorage.setItem("msanii_pending_org_invite", "{not json");
    expect(readPendingInvite()).toBeNull();
    sessionStorage.setItem("msanii_pending_org_invite", JSON.stringify({ accepted: true }));
    expect(readPendingInvite()).toBeNull();
  });
});
