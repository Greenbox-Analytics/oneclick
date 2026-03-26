export interface AudioFolder {
  id: string;
  artist_id: string;
  name: string;
  parent_id: string | null;
  created_at: string;
}

export interface AudioFile {
  id: string;
  folder_id: string;
  file_name: string;
  file_url: string;
  file_path: string;
  file_size: number | null;
  file_type: string | null;
  created_at: string;
}

export interface ProjectAudioLink {
  project_id: string;
  audio_file_id: string;
  created_at: string;
}
