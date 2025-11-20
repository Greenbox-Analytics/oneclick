-- Create artists table
CREATE TABLE public.artists (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  bio TEXT,
  genres TEXT[] DEFAULT '{}',
  avatar_url TEXT,
  social_instagram TEXT,
  social_tiktok TEXT,
  social_youtube TEXT,
  dsp_spotify TEXT,
  dsp_apple_music TEXT,
  dsp_soundcloud TEXT,
  additional_epk TEXT,
  additional_press_kit TEXT,
  additional_linktree TEXT,
  custom_links JSONB DEFAULT '[]',
  has_contract BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.artists ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (adjust based on your needs)
CREATE POLICY "Anyone can view artists" 
ON public.artists 
FOR SELECT 
USING (true);

CREATE POLICY "Anyone can create artists" 
ON public.artists 
FOR INSERT 
WITH CHECK (true);

CREATE POLICY "Anyone can update artists" 
ON public.artists 
FOR UPDATE 
USING (true);

CREATE POLICY "Anyone can delete artists" 
ON public.artists 
FOR DELETE 
USING (true);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = public;

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_artists_updated_at
BEFORE UPDATE ON public.artists
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();