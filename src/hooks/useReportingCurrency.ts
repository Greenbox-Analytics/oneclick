import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";

const DEFAULT_CURRENCY = "USD";

/**
 * Persists the reporting currency selection to localStorage,
 * namespaced per-account so different users on the same browser
 * each get their own preference.
 *
 * Returns [currency, setCurrency] — a stable pair that survives
 * navigation (components that call this hook all share the same
 * stored value once they read from localStorage on mount).
 */
export function useReportingCurrency(): [string, (c: string) => void] {
  const { user } = useAuth();
  const storageKey = `reporting-currency:${user?.id ?? "anon"}`;

  const [currency, setCurrencyState] = useState<string>(() => {
    // Guard for SSR / environments where localStorage is absent
    try {
      return localStorage.getItem(storageKey) ?? DEFAULT_CURRENCY;
    } catch {
      return DEFAULT_CURRENCY;
    }
  });

  // Re-read from localStorage when the user id changes (e.g. account switch)
  useEffect(() => {
    try {
      const stored = localStorage.getItem(storageKey);
      setCurrencyState(stored ?? DEFAULT_CURRENCY);
    } catch {
      setCurrencyState(DEFAULT_CURRENCY);
    }
  }, [storageKey]);

  const setCurrency = (c: string) => {
    setCurrencyState(c);
    try {
      localStorage.setItem(storageKey, c);
    } catch {
      // localStorage unavailable — state update still applied in-memory
    }
  };

  return [currency, setCurrency];
}
