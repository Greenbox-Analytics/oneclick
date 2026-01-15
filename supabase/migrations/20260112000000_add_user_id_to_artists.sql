-- Add user_id column to artists table
ALTER TABLE public.artists 
ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;

-- Drop existing policies
DROP POLICY IF EXISTS "Anyone can view artists" ON public.artists;
DROP POLICY IF EXISTS "Anyone can create artists" ON public.artists;
DROP POLICY IF EXISTS "Anyone can update artists" ON public.artists;
DROP POLICY IF EXISTS "Anyone can delete artists" ON public.artists;

-- Create new policies that restrict access to user's own artists
CREATE POLICY "Users can view their own artists" 
ON public.artists 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own artists" 
ON public.artists 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own artists" 
ON public.artists 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own artists" 
ON public.artists 
FOR DELETE 
USING (auth.uid() = user_id);

-- Create index for better query performance
CREATE INDEX idx_artists_user_id ON public.artists(user_id);
