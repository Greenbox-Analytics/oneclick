import { useState, useMemo, useCallback } from "react";
import {
  startOfWeek,
  endOfWeek,
  startOfMonth,
  endOfMonth,
  addWeeks,
  subWeeks,
  addMonths,
  subMonths,
  addDays,
  subDays,
  format,
  isSameDay,
} from "date-fns";

type BoardPeriod = "weekly" | "biweekly" | "monthly" | "custom";

interface UseBoardPeriodOptions {
  boardPeriod: BoardPeriod;
  customPeriodDays?: number;
}

// Fixed epoch for biweekly/custom alignment
const EPOCH = new Date(2024, 0, 1); // Jan 1 2024

function getCustomPeriodStart(current: Date, periodDays: number): Date {
  const diffMs = current.getTime() - EPOCH.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  const periodsElapsed = Math.floor(diffDays / periodDays);
  return addDays(EPOCH, periodsElapsed * periodDays);
}

export function useBoardPeriod({ boardPeriod, customPeriodDays = 14 }: UseBoardPeriodOptions) {
  const [referenceDate, setReferenceDate] = useState(new Date());

  const { periodStart, periodEnd } = useMemo(() => {
    let start: Date;
    let end: Date;

    switch (boardPeriod) {
      case "weekly":
        start = startOfWeek(referenceDate);
        end = endOfWeek(referenceDate);
        break;
      case "biweekly": {
        start = getCustomPeriodStart(referenceDate, 14);
        end = addDays(start, 13);
        break;
      }
      case "monthly":
        start = startOfMonth(referenceDate);
        end = endOfMonth(referenceDate);
        break;
      case "custom": {
        start = getCustomPeriodStart(referenceDate, customPeriodDays);
        end = addDays(start, customPeriodDays - 1);
        break;
      }
      default:
        start = startOfMonth(referenceDate);
        end = endOfMonth(referenceDate);
    }

    return {
      periodStart: format(start, "yyyy-MM-dd"),
      periodEnd: format(end, "yyyy-MM-dd"),
    };
  }, [referenceDate, boardPeriod, customPeriodDays]);

  const periodLabel = useMemo(() => {
    switch (boardPeriod) {
      case "weekly":
        return `Week of ${format(new Date(periodStart + "T12:00:00"), "MMM d, yyyy")}`;
      case "biweekly":
        return `${format(new Date(periodStart + "T12:00:00"), "MMM d")} – ${format(new Date(periodEnd + "T12:00:00"), "MMM d, yyyy")}`;
      case "monthly":
        return format(new Date(periodStart + "T12:00:00"), "MMMM yyyy");
      case "custom":
        return `${format(new Date(periodStart + "T12:00:00"), "MMM d")} – ${format(new Date(periodEnd + "T12:00:00"), "MMM d, yyyy")}`;
      default:
        return format(new Date(periodStart + "T12:00:00"), "MMMM yyyy");
    }
  }, [boardPeriod, periodStart, periodEnd]);

  const isCurrentPeriod = useMemo(() => {
    const now = new Date();
    const start = new Date(periodStart + "T00:00:00");
    const end = new Date(periodEnd + "T23:59:59");
    return now >= start && now <= end;
  }, [periodStart, periodEnd]);

  const goToPrevPeriod = useCallback(() => {
    setReferenceDate((prev) => {
      switch (boardPeriod) {
        case "weekly": return subWeeks(prev, 1);
        case "biweekly": return subWeeks(prev, 2);
        case "monthly": return subMonths(prev, 1);
        case "custom": return subDays(prev, customPeriodDays);
        default: return subMonths(prev, 1);
      }
    });
  }, [boardPeriod, customPeriodDays]);

  const goToNextPeriod = useCallback(() => {
    setReferenceDate((prev) => {
      switch (boardPeriod) {
        case "weekly": return addWeeks(prev, 1);
        case "biweekly": return addWeeks(prev, 2);
        case "monthly": return addMonths(prev, 1);
        case "custom": return addDays(prev, customPeriodDays);
        default: return addMonths(prev, 1);
      }
    });
  }, [boardPeriod, customPeriodDays]);

  const goToCurrentPeriod = useCallback(() => {
    setReferenceDate(new Date());
  }, []);

  return {
    periodStart,
    periodEnd,
    periodLabel,
    isCurrentPeriod,
    goToPrevPeriod,
    goToNextPeriod,
    goToCurrentPeriod,
  };
}
