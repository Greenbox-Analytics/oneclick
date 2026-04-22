CREATE TABLE IF NOT EXISTS public.mailing_list (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  name TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.mailing_list ENABLE ROW LEVEL SECURITY;

-- Allow anyone to insert (for the public signup form)
CREATE POLICY "Allow anonymous inserts" ON public.mailing_list
  FOR INSERT WITH CHECK (true);
