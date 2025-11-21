-- Create projects table
CREATE TABLE public.projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  artist_id UUID REFERENCES public.artists(id) ON DELETE CASCADE NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- Enable RLS
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Anyone can view projects"
ON public.projects
FOR SELECT
USING (true);

CREATE POLICY "Anyone can create projects"
ON public.projects
FOR INSERT
WITH CHECK (true);

CREATE POLICY "Anyone can update projects"
ON public.projects
FOR UPDATE
USING (true);

CREATE POLICY "Anyone can delete projects"
ON public.projects
FOR DELETE
USING (true);

-- Create trigger for updated_at
CREATE TRIGGER update_projects_updated_at
BEFORE UPDATE ON public.projects
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create project_files table for folder organization
CREATE TABLE public.project_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID REFERENCES public.projects(id) ON DELETE CASCADE NOT NULL,
  folder_category TEXT NOT NULL CHECK (folder_category IN ('contracts', 'split_sheets', 'royalty_statements', 'other_files')),
  file_name TEXT NOT NULL,
  file_url TEXT NOT NULL,
  file_size BIGINT,
  file_type TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- Enable RLS
ALTER TABLE public.project_files ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Anyone can view project files"
ON public.project_files
FOR SELECT
USING (true);

CREATE POLICY "Anyone can create project files"
ON public.project_files
FOR INSERT
WITH CHECK (true);

CREATE POLICY "Anyone can update project files"
ON public.project_files
FOR UPDATE
USING (true);

CREATE POLICY "Anyone can delete project files"
ON public.project_files
FOR DELETE
USING (true);

-- Create trigger for updated_at
CREATE TRIGGER update_project_files_updated_at
BEFORE UPDATE ON public.project_files
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();