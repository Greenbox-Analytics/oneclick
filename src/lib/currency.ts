// Keep in sync with CURRENCY_SYMBOLS / fmt_money in src/backend/expenses/common.py
export const CURRENCY_SYMBOLS: Record<string, string> = {
  USD: "US$",
  EUR: "€",
  CAD: "CA$",
  AUD: "A$",
  GBP: "£",
};

/** Display symbol for a currency code (case-insensitive); undefined if unknown. */
export const currencySymbol = (currency: string = "USD"): string | undefined =>
  CURRENCY_SYMBOLS[(currency || "USD").toUpperCase()];

/** Format an amount with its currency symbol, e.g. "US$1,234.50". */
export const formatCurrency = (n: number, currency: string = "USD"): string => {
  const symbol = currencySymbol(currency);
  const num = (n || 0).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return symbol ? `${symbol}${num}` : `${num} ${(currency || "").toUpperCase()}`;
};
