"""One-time backfill: fix Content-Type on storage objects uploaded as text/plain.

Backend uploads historically omitted content-type, so the storage client
defaulted to text/plain and browsers refuse to render those PDFs inline
("Failed to load PDF document" in the contract slide-over).

The SQL migration (20260610000000_fix_storage_object_mimetypes.sql) updated
storage.objects.metadata, but Supabase Storage serves Content-Type from the
underlying S3 object, so the only real fix is a re-upload in place (same
bytes, correct content-type). Because the migration already rewrote the DB
metadata, affected objects are detected by checking the *served* header via
a signed URL, not by listing metadata.

Idempotent: objects already serving the right Content-Type are skipped.

Usage (from src/backend/):
    poetry run python -m scripts.backfill_storage_mimetypes --dry-run
    poetry run python -m scripts.backfill_storage_mimetypes
"""

import argparse
import os
import sys
import urllib.request

from dotenv import load_dotenv
from supabase import create_client

BUCKET = "project-files"

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

MAGIC_BYTES = {
    ".pdf": b"%PDF-",
    ".xlsx": b"PK\x03\x04",  # OOXML files are zip archives
    ".docx": b"PK\x03\x04",
}


def walk(bucket, prefix=""):
    """Yield full paths of all objects in the bucket (folders have no metadata)."""
    for item in bucket.list(prefix, {"limit": 1000}):
        full = f"{prefix}/{item['name']}" if prefix else item["name"]
        if item.get("id") is None and item.get("metadata") is None:
            yield from walk(bucket, full)
        else:
            yield full


def served_content_type(bucket, path) -> str:
    url = bucket.create_signed_url(path, 120)["signedURL"]
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        return (resp.headers.get("content-type") or "").split(";")[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without re-uploading")
    args = parser.parse_args()

    load_dotenv("../../.env")
    load_dotenv()
    url = os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("VITE_SUPABASE_SECRET_KEY")
    if not url or not key:
        print("ERROR: VITE_SUPABASE_URL / VITE_SUPABASE_SECRET_KEY not set")
        return 1

    bucket = create_client(url, key).storage.from_(BUCKET)

    checked = fixed = skipped = errors = 0
    for path in walk(bucket):
        ext = os.path.splitext(path)[1].lower()
        expected = CONTENT_TYPES.get(ext)
        if not expected:
            continue
        checked += 1
        try:
            current = served_content_type(bucket, path)
            if current == expected:
                skipped += 1
                continue
            if args.dry_run:
                print(f"WOULD FIX {path}  ({current} -> {expected})")
                fixed += 1
                continue

            data = bucket.download(path)
            magic = MAGIC_BYTES[ext]
            if not data or not data.startswith(magic):
                print(f"SKIP (content mismatch, not touching) {path}")
                errors += 1
                continue

            bucket.upload(path, data, file_options={"content-type": expected, "upsert": "true"})

            after = served_content_type(bucket, path)
            if after != expected:
                print(f"WARNING: still serving {after} after re-upload: {path}")
                errors += 1
            else:
                print(f"FIXED {path}  ({current} -> {expected})")
                fixed += 1
        except Exception as e:  # noqa: BLE001 — keep going, report at the end
            print(f"ERROR {path}: {e}")
            errors += 1

    label = "would fix" if args.dry_run else "fixed"
    print(f"\nChecked {checked} | {label} {fixed} | already correct {skipped} | errors {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
