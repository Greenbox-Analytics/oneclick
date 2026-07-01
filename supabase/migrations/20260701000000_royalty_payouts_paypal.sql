-- PayPal payment support on royalty_payouts.
--
-- Payouts keep the existing draft -> paid state machine: a payout stays
-- 'draft' until the backend captures the approved PayPal order, then flips
-- to 'paid' via the same update path as manual "Mark paid".
--
-- payment_method records HOW a payout was settled ('manual' | 'paypal').
-- paypal_order_id is set when a checkout order is created (may point to an
-- abandoned/expired order until capture succeeds); paypal_capture_id is set
-- only after a COMPLETED capture.
--
-- No RLS changes needed — new columns inherit the existing row policies.

alter table public.royalty_payouts
  add column payment_method text not null default 'manual'
    check (payment_method in ('manual', 'paypal')),
  add column paypal_order_id text,
  add column paypal_capture_id text;
