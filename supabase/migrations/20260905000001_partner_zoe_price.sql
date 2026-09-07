-- Price row for Zoe on the partner API (POST /zoe/v1/chat/completions).
-- Independent of the product's zoe_message so the two can be dialled apart,
-- exactly like partner_oneclick_run vs oneclick_run. Seeded at the product's
-- base (5). Public-read like every credit_prices row; NOT added to
-- Entitlements.to_dict()'s hand-built prices block (no client renders it).
INSERT INTO credit_prices (action, credits) VALUES ('partner_zoe_message', 5)
ON CONFLICT (action) DO NOTHING;
