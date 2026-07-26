-- Multi-currency expenses: store the currency the expense was entered in.
-- Aggregation (tracker totals, exports, OneClick net deductions) converts to USD
-- at read time via the Bank of Canada FX system; the stored amount is never mutated.
ALTER TABLE project_expenses
  ADD COLUMN currency TEXT NOT NULL DEFAULT 'USD'
  CHECK (currency IN ('USD', 'EUR', 'CAD', 'AUD'));
