"""Pydantic models for Dropbox integration."""

from pydantic import BaseModel


class DropboxImportRequest(BaseModel):
    dropbox_file_id: str  # Dropbox file id ("id:...")
    project_id: str
    file_type: str | None = "contract"  # folder_category for the project_files row


class DropboxExportRequest(BaseModel):
    project_file_id: str
    dropbox_folder_id: str | None = None  # None/"" = root, else a folder id ("id:...")
