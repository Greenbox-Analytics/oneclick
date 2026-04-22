-- Create the audio-files storage bucket (private) and its RLS policies.
-- The audio_files / audio_folders tables already gate metadata by artist
-- ownership, so anyone who can read those rows is authorized for the object.
-- Access to objects is issued via signed URLs at read time.

INSERT INTO storage.buckets (id, name, public)
VALUES ('audio-files', 'audio-files', false)
ON CONFLICT (id) DO NOTHING;

CREATE POLICY "Authenticated users can upload audio files"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'audio-files');

CREATE POLICY "Authenticated users can read audio files"
ON storage.objects FOR SELECT
TO authenticated
USING (bucket_id = 'audio-files');

CREATE POLICY "Authenticated users can update audio files"
ON storage.objects FOR UPDATE
TO authenticated
USING (bucket_id = 'audio-files');

CREATE POLICY "Authenticated users can delete audio files"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'audio-files');
