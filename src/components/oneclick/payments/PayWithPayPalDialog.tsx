// src/components/oneclick/payments/PayWithPayPalDialog.tsx
import { useState } from "react";
import { CheckCheck, Info, Loader2, TriangleAlert } from "lucide-react";
import { PayPalScriptProvider, PayPalButtons } from "@paypal/react-paypal-js";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ApiError } from "@/lib/apiFetch";
import {
  useCreatePaypalOrder,
  useCapturePaypalOrder,
  useSetPayeeCurrency,
} from "@/hooks/useRoyalties";
import type { PayoutOut, PayeeSummary } from "@/hooks/useRoyalties";
import { PartyAvatar, fmtMoney } from "./shared";
import { useToast } from "@/hooks/use-toast";

/** Mirror of PAYPAL_SUPPORTED_CURRENCIES in src/backend/oneclick/royalties/paypal_client.py. */
export const PAYPAL_SUPPORTED_CURRENCIES = new Set([
  "AUD", "CAD", "CHF", "CZK", "DKK", "EUR", "GBP", "HKD", "HUF", "ILS", "JPY",
  "MXN", "NOK", "NZD", "PHP", "PLN", "SEK", "SGD", "THB", "TWD", "USD",
]);

/** PayPal payments are available only when the public client id is configured. */
export function isPaypalEnabled(): boolean {
  return !!import.meta.env.VITE_PAYPAL_CLIENT_ID;
}

interface PayWithPayPalDialogProps {
  payout: PayoutOut;
  payee?: PayeeSummary;
  onClose: () => void;
}

export function PayWithPayPalDialog({ payout, payee, onClose }: PayWithPayPalDialogProps) {
  const { toast } = useToast();
  const createOrder = useCreatePaypalOrder();
  const captureOrder = useCapturePaypalOrder();
  const patchPayee = useSetPayeeCurrency();

  const [phase, setPhase] = useState<"form" | "capturing" | "success">("form");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [emailDraft, setEmailDraft] = useState("");
  // Set once the user saves an email inline, so we can proceed without
  // waiting for the payees query to refetch.
  const [savedEmail, setSavedEmail] = useState<string | null>(null);

  const payeeName = payee?.display_name ?? "this payee";
  const payeeEmail = savedEmail ?? payee?.email?.trim() ?? "";
  const currency = payout.pay_currency?.toUpperCase() ?? "";
  const currencySupported = PAYPAL_SUPPORTED_CURRENCIES.has(currency);
  const clientId = import.meta.env.VITE_PAYPAL_CLIENT_ID as string;

  const handleSaveEmail = () => {
    const email = emailDraft.trim();
    patchPayee.mutate(
      { id: payout.payee_id, email },
      {
        onSuccess: () => setSavedEmail(email),
        onError: () =>
          toast({ variant: "destructive", title: "Couldn't save the email. Please try again." }),
      },
    );
  };

  const handleApprove = async () => {
    setPhase("capturing");
    try {
      await captureOrder.mutateAsync(payout.id);
      setPhase("success");
    } catch (err) {
      setPhase("form");
      setErrorMsg(
        err instanceof ApiError && err.status === 400
          ? err.message
          : "The payment didn't go through — nothing was sent. You can try again.",
      );
    }
  };

  return (
    <Dialog open onOpenChange={(o) => { if (!o && phase !== "capturing") onClose(); }}>
      <DialogContent className="max-w-md gap-0 overflow-y-auto p-0 max-h-[90vh]">
        <DialogHeader className="border-b border-border px-5 py-4">
          <DialogTitle>{phase === "success" ? "Payment sent" : "Pay with PayPal"}</DialogTitle>
        </DialogHeader>

        {phase === "success" ? (
          <div className="px-6 py-8 text-center">
            <div className="mx-auto mb-3.5 flex h-[60px] w-[60px] items-center justify-center rounded-full bg-[hsl(var(--pay-paid-bg))] text-[hsl(var(--pay-paid-fg))]">
              <CheckCheck className="h-7 w-7" />
            </div>
            <div className="text-base font-bold">
              Paid {fmtMoney(payout.total_amount, currency)} to {payeeName}
            </div>
            <p className="mx-auto mt-2 max-w-[42ch] text-[13px] text-muted-foreground">
              The money is on its way to <strong>{payeeEmail}</strong> via PayPal. This payout is
              now marked as paid.
            </p>
            <div className="mt-6 flex justify-end">
              <Button onClick={onClose}>Done</Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-4 px-5 py-5">
            {/* Payee + amount summary */}
            <div className="flex items-center justify-between rounded-xl border border-border bg-card p-3.5">
              <span className="flex min-w-0 items-center gap-2.5">
                <PartyAvatar id={payout.payee_id} name={payeeName} size={34} />
                <span className="min-w-0">
                  <div className="text-[14px] font-semibold">{payeeName}</div>
                  {payeeEmail && (
                    <div className="truncate text-[12px] text-muted-foreground">{payeeEmail}</div>
                  )}
                </span>
              </span>
              <span className="text-right">
                <div className="font-mono text-[17px] font-bold tabular-nums tracking-tight">
                  {fmtMoney(payout.total_amount, currency)}
                </div>
                <div className="text-[11.5px] text-muted-foreground">{currency}</div>
              </span>
            </div>

            {!currencySupported ? (
              <div className="flex items-start gap-2 rounded-[10px] border border-[hsl(var(--pay-out-bg))] bg-[hsl(var(--pay-out-bg))] px-3 py-2.5 text-[12.5px] leading-relaxed text-[hsl(var(--pay-out-fg))]">
                <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                <span>
                  PayPal can't send {currency}. To pay {payeeName} through PayPal, cancel this
                  draft, change their payout currency in the Parties tab, and create a new payout.
                </span>
              </div>
            ) : !payeeEmail ? (
              <div className="flex flex-col gap-2.5">
                <div className="flex items-start gap-2 rounded-[10px] border border-[hsl(217_70%_88%)] bg-[hsl(217_80%_95%)] px-3 py-2.5 text-[12.5px] leading-relaxed text-[hsl(217_50%_32%)] dark:border-[hsl(217_40%_28%)] dark:bg-[hsl(217_45%_18%)] dark:text-[hsl(217_80%_80%)]">
                  <Info className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>
                    To pay {payeeName} through PayPal, add the email address linked to their PayPal
                    account.
                  </span>
                </div>
                <div className="flex gap-2">
                  <Input
                    type="email"
                    placeholder="payee@example.com"
                    value={emailDraft}
                    onChange={(e) => setEmailDraft(e.target.value)}
                  />
                  <Button
                    disabled={!/^\S+@\S+\.\S+$/.test(emailDraft.trim()) || patchPayee.isPending}
                    onClick={handleSaveEmail}
                  >
                    {patchPayee.isPending ? "Saving…" : "Save"}
                  </Button>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-start gap-2 rounded-[10px] border border-[hsl(217_70%_88%)] bg-[hsl(217_80%_95%)] px-3 py-2.5 text-[12.5px] leading-relaxed text-[hsl(217_50%_32%)] dark:border-[hsl(217_40%_28%)] dark:bg-[hsl(217_45%_18%)] dark:text-[hsl(217_80%_80%)]">
                  <Info className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>
                    The payment will go to <strong>{payeeEmail}</strong>. Double-check this is{" "}
                    {payeeName}'s PayPal email — money sent to the wrong address is hard to get
                    back. You'll log in to your own PayPal account to approve it.
                  </span>
                </div>

                {errorMsg && (
                  <div className="flex items-start gap-2 rounded-[10px] border border-destructive/30 bg-destructive/10 px-3 py-2.5 text-[12.5px] leading-relaxed text-destructive">
                    <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                    <span>{errorMsg}</span>
                  </div>
                )}

                {phase === "capturing" ? (
                  <div className="flex items-center justify-center gap-2 py-4 text-[13px] text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" /> Finishing your payment — don't
                    close this window…
                  </div>
                ) : (
                  <PayPalScriptProvider
                    options={{ clientId, currency, intent: "capture" }}
                  >
                    <PayPalButtons
                      style={{ layout: "vertical", label: "pay" }}
                      forceReRender={[payout.id, currency]}
                      createOrder={async () => {
                        setErrorMsg(null);
                        try {
                          const res = await createOrder.mutateAsync(payout.id);
                          return res.paypal_order_id;
                        } catch (err) {
                          setErrorMsg(
                            err instanceof ApiError && (err.status === 400 || err.status === 409)
                              ? err.message
                              : "Couldn't start the PayPal payment. Please try again.",
                          );
                          throw err;
                        }
                      }}
                      onApprove={handleApprove}
                      onError={() => {
                        setErrorMsg((msg) => msg ?? "The payment didn't go through — nothing was sent. You can try again.");
                      }}
                    />
                  </PayPalScriptProvider>
                )}
              </>
            )}

            {phase !== "capturing" && (
              <div className="flex justify-start">
                <Button variant="outline" onClick={onClose}>
                  Cancel
                </Button>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
