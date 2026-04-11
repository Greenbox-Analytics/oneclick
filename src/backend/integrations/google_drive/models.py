"""Pydantic models for Google Drive integration."""

from pydantic import BaseModel


class DriveImportRequest(BaseModel):
    drive_file_id: str
    project_id: str
    file_type: str | None = "contract"  # contract or royalty_statement


class DriveExportRequest(BaseModel):
    project_file_id: str
    drive_folder_id: str | None = None  # None = root


class DriveSyncSetup(BaseModel):
    project_id: str
    drive_folder_id: str
    sync_direction: str = "bidirectional"  # to_drive, from_drive, bidirectional
