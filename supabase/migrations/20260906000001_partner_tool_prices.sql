-- Price rows for the two tools the partner API gained on 2026-09-06:
--   POST /registry/v1/contract-terms  -> partner_registry_parse (product: registry_parse, 30)
--   POST /splitsheet/v1/documents     -> partner_split_sheet    (product: split_sheet, 20)
-- Independent of the product rows so the two can be dialled apart, exactly
-- like partner_oneclick_run / partner_zoe_message. Seeded at the product's
-- base. Public-read like every credit_prices row; NOT added to
-- Entitlements.to_dict()'s hand-built prices block (no client renders them).
INSERT INTO credit_prices (action, credits) VALUES
  ('partner_registry_parse', 30),
  ('partner_split_sheet', 20)
ON CONFLICT (action) DO NOTHING;
