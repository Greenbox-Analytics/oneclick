"""Pydantic models for Google Drive integration."""

from pydantic import BaseModel
from typing import Optional


class DriveImportRequest(BaseModel):
    drive_file_id: str
    project_id: str
    file_type: Optional[str] = "contract"  # contract or royalty_statement


class DriveExportRequest(BaseModel):
    project_file_id: str
    drive_folder_id: Optional[str] = None  # None = root


class DriveSyncSetup(BaseModel):
    project_id: str
    drive_folder_id: str
    sync_direction: str = "bidirectional"  # to_drive, from_drive, bidirectional
