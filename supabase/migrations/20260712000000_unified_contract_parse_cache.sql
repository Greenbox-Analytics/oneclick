-- Unify contract_parse_cache to store the FULL parsed ContractData (parties, works,
-- royalty_shares incl. basis/terms, default_basis) instead of Registry's lossy pivoted
-- splits shape. Keyed by (content_hash, parser_version): bumping the extraction prompt
-- (PARSER_PROMPT_VERSION) or the OpenAI model (OPENAI_LLM_MODEL_LARGE) changes
-- parser_version, so stale entries are ignored rather than served.
--
-- NOTE: `content_hash` now holds the SHA-256 of the CANONICAL parse text (the contract
-- markdown with [[PAGE n]] markers stripped) — i.e. the exact text fed to the extractor —
-- NOT the source file bytes. Keying on the parser input makes each entry single-valued:
-- callers that derive different markdown from the same PDF get different keys and can
-- never serve each other a mismatched parse. (Column name kept for continuity.)
--
-- The `parsed` payload semantics change, so existing rows are incompatible with the new
-- reader. Wiping them is safe: it is a cache and repopulates on next parse.
--
-- RLS: the table already has RLS enabled with NO policies (service-role only); nothing
-- below re-grants client access, so it stays backend-only.

truncate table contract_parse_cache;

alter table contract_parse_cache
  add column if not exists parser_version text not null default 'legacy';

alter table contract_parse_cache drop constraint if exists contract_parse_cache_pkey;
alter table contract_parse_cache add primary key (content_hash, parser_version);
