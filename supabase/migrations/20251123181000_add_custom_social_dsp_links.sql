-- Add custom social media and DSP links columns to artists table

-- 1. Add custom_social_links column for storing additional social media platforms
ALTER TABLE public.artists
ADD COLUMN IF NOT EXISTS custom_social_links JSONB DEFAULT '[]';

-- 2. Add custom_dsp_links column for storing additional streaming platforms
ALTER TABLE public.artists
ADD COLUMN IF NOT EXISTS custom_dsp_links JSONB DEFAULT '[]';

-- 3. Add comments for documentation
COMMENT ON COLUMN public.artists.custom_social_links IS 
'Custom social media links (e.g., Twitter, Threads, etc.) stored as JSON array: [{"id": "1", "label": "Twitter", "url": "https://twitter.com/username"}]';

COMMENT ON COLUMN public.artists.custom_dsp_links IS 
'Custom DSP/streaming platform links (e.g., Tidal, Deezer, etc.) stored as JSON array: [{"id": "1", "label": "Tidal", "url": "https://tidal.com/artist"}]';
