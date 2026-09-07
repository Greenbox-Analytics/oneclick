import { beforeEach, describe, expect, it } from "vitest";
import {
  clearPendingInvite,
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

describe("pending invite stash", () => {
  beforeEach(() => sessionStorage.clear());

  it("is empty by default", () => {
    expect(readPendingInvite()).toBeNull();
  });

  it("stores and reads an unaccepted invite", () => {
    stashPendingInvite({ token: "tok", accepted: false });
    expect(readPendingInvite()).toEqual({ token: "tok", accepted: false, orgName: null, kind: null });
  });

  it("stores the org name once accepted", () => {
    stashPendingInvite({ token: "tok", accepted: true, orgName: "Acme", kind: "self_serve" });
    expect(readPendingInvite()).toEqual({ token: "tok", accepted: true, orgName: "Acme", kind: "self_serve" });
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
