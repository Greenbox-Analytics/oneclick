import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { DayPicker } from "react-day-picker";

import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";

export type CalendarProps = React.ComponentProps<typeof DayPicker>;

// react-day-picker v9+ renamed its classNames/components API (head_row → weekdays,
// cell → day, day → day_button, IconLeft/IconRight → Chevron, etc.). These class names
// target that API. The app does not import react-day-picker's default stylesheet, so the
// layout is driven entirely by the Tailwind classes below.
function Calendar({ className, classNames, showOutsideDays = true, ...props }: CalendarProps) {
  return (
    <DayPicker
      showOutsideDays={showOutsideDays}
      className={cn("p-3", className)}
      classNames={{
        months: "relative flex flex-col gap-4 sm:flex-row sm:gap-4",
        month: "space-y-4",
        month_caption: "flex h-7 items-center justify-center pt-1",
        caption_label: "text-sm font-medium",
        // Nav is a sibling of the month(s) inside the relative `months` container, so it
        // overlays the caption row with the prev/next buttons pinned to the edges.
        nav: "absolute inset-x-0 top-0 flex items-center justify-between px-1",
        button_previous: cn(
          buttonVariants({ variant: "outline" }),
          "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100",
        ),
        button_next: cn(
          buttonVariants({ variant: "outline" }),
          "h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100",
        ),
        month_grid: "w-full border-collapse space-y-1",
        weekdays: "flex",
        weekday: "w-9 rounded-md text-[0.8rem] font-normal text-muted-foreground",
        week: "mt-2 flex w-full",
        day: cn(
          "relative h-9 w-9 p-0 text-center text-sm focus-within:relative focus-within:z-20",
          "[&:has([aria-selected])]:bg-accent [&:has([aria-selected].day-outside)]:bg-accent/50",
          "first:[&:has([aria-selected])]:rounded-l-md last:[&:has([aria-selected])]:rounded-r-md",
          "[&:has([aria-selected].day-range-end)]:rounded-r-md",
        ),
        day_button: cn(
          buttonVariants({ variant: "ghost" }),
          "h-9 w-9 p-0 font-normal",
          "aria-selected:bg-primary aria-selected:text-primary-foreground",
          "aria-selected:hover:bg-primary aria-selected:hover:text-primary-foreground",
          "aria-selected:focus:bg-primary aria-selected:focus:text-primary-foreground",
        ),
        range_end: "day-range-end",
        range_middle: "aria-selected:bg-accent aria-selected:text-accent-foreground",
        today: "[&>button]:bg-accent [&>button]:text-accent-foreground",
        outside: "day-outside text-muted-foreground opacity-50",
        disabled: "text-muted-foreground opacity-50 [&>button]:pointer-events-none",
        hidden: "invisible",
        ...classNames,
      }}
      components={{
        Chevron: ({ orientation, className: chevronClassName, ...chevronProps }) => {
          const Icon = orientation === "left" ? ChevronLeft : ChevronRight;
          return <Icon className={cn("h-4 w-4", chevronClassName)} {...chevronProps} />;
        },
      }}
      {...props}
    />
  );
}
Calendar.displayName = "Calendar";

export { Calendar };
