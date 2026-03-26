-- Secure projects table: scope access to users who own the artist
-- Projects link to users via: projects.artist_id -> artists.user_id

-- Drop existing permissive policies
DROP POLICY IF EXISTS "Anyone can view projects" ON public.projects;
DROP POLICY IF EXISTS "Anyone can create projects" ON public.projects;
DROP POLICY IF EXISTS "Anyone can update projects" ON public.projects;
DROP POLICY IF EXISTS "Anyone can delete projects" ON public.projects;

-- Create user-scoped policies
CREATE POLICY "Users can view own projects"
ON public.projects FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.artists
    WHERE artists.id = projects.artist_id
    AND artists.user_id = auth.uid()
  )
);

CREATE POLICY "Users can create own projects"
ON public.projects FOR INSERT
WITH CHECK (
  EXISTS (
    SELECT 1 FROM public.artists
    WHERE artists.id = artist_id
    AND artists.user_id = auth.uid()
  )
);

CREATE POLICY "Users can update own projects"
ON public.projects FOR UPDATE
USING (
  EXISTS (
    SELECT 1 FROM public.artists
    WHERE artists.id = projects.artist_id
    AND artists.user_id = auth.uid()
  )
);

CREATE POLICY "Users can delete own projects"
ON public.projects FOR DELETE
USING (
  EXISTS (
    SELECT 1 FROM public.artists
    WHERE artists.id = projects.artist_id
    AND artists.user_id = auth.uid()
  )
);

-- Secure project_files table: scope access through projects -> artists -> users
-- project_files link to users via: project_files.project_id -> projects.artist_id -> artists.user_id

-- Drop existing permissive policies
DROP POLICY IF EXISTS "Anyone can view project files" ON public.project_files;
DROP POLICY IF EXISTS "Anyone can create project files" ON public.project_files;
DROP POLICY IF EXISTS "Anyone can update project files" ON public.project_files;
DROP POLICY IF EXISTS "Anyone can delete project files" ON public.project_files;

-- Create user-scoped policies
CREATE POLICY "Users can view own project files"
ON public.project_files FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.projects
    JOIN public.artists ON artists.id = projects.artist_id
    WHERE projects.id = project_files.project_id
    AND artists.user_id = auth.uid()
  )
);

CREATE POLICY "Users can create own project files"
ON public.project_files FOR INSERT
WITH CHECK (
  EXISTS (
    SELECT 1 FROM public.projects
    JOIN public.artists ON artists.id = projects.artist_id
    WHERE projects.id = project_id
    AND artists.user_id = auth.uid()
  )
);

CREATE POLICY "Users can update own project files"
ON public.project_files FOR UPDATE
USING (
  EXISTS (
    SELECT 1 FROM public.projects
    JOIN public.artists ON artists.id = projects.artist_id
    WHERE projects.id = project_files.project_id
    AND artists.user_id = auth.uid()
  )
);

CREATE POLICY "Users can delete own project files"
ON public.project_files FOR DELETE
USING (
  EXISTS (
    SELECT 1 FROM public.projects
    JOIN public.artists ON artists.id = projects.artist_id
    WHERE projects.id = project_files.project_id
    AND artists.user_id = auth.uid()
  )
);
