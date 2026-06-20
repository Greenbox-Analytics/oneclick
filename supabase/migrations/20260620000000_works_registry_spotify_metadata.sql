-- Add genre, label, and credited-artist columns to works_registry so the
-- "Pull from Spotify" flow can persist metadata it already fetches but
-- previously dropped. Genre and label were returned by the Spotify service but
-- had no destination column; featured_artists captures the track's credited
-- artists (main + featured) as display-only metadata, kept separate from the
-- ownership_stakes / royalty splits system.

alter table works_registry
  add column if not exists genre text,
  add column if not exists label text,
  add column if not exists featured_artists jsonb not null default '[]'::jsonb;
