import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Music, Plus, Search, Trash2, Pencil, Mail, Phone, Building2, DollarSign, ArrowLeft, BookOpen } from "lucide-react";
import { useNavigate } from "react-router-dom";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { ContactFormDialog } from "@/components/ContactFormDialog";
import { PaymentDialog } from "@/components/PaymentDialog";
import type { Tables } from "@/integrations/supabase/types";

const Contacts = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [contacts, setContacts] = useState<Tables<"contacts">[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [contactToDelete, setContactToDelete] = useState<Tables<"contacts"> | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [editingContact, setEditingContact] = useState<Tables<"contacts"> | null>(null);
  const [paymentContact, setPaymentContact] = useState<Tables<"contacts"> | null>(null);

  const fetchContacts = async () => {
    if (!user) {
      setIsLoading(false);
      return;
    }

    const { data, error } = await supabase
      .from("contacts")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      console.error("Error fetching contacts:", error);
      toast({
        title: "Error",
        description: "Failed to load contacts",
        variant: "destructive",
      });
    } else if (data) {
      setContacts(data);
    }
    setIsLoading(false);
  };

  useEffect(() => {
    fetchContacts();
  }, [user]);

  const handleSave = async (data: {
    name: string;
    email: string;
    phone: string;
    role: string;
    notes: string;
    bank_country: string;
    bank_account_holder: string;
    bank_transit_number: string;
    bank_institution_number: string;
    bank_account_number: string;
    bank_iban: string;
    bank_swift_bic: string;
    bank_currency: string;
  }) => {
    if (!user) return;

    const contactData = {
      name: data.name,
      email: data.email || null,
      phone: data.phone || null,
      role: data.role || null,
      notes: data.notes || null,
      bank_country: data.bank_country || null,
      bank_account_holder: data.bank_account_holder || null,
      bank_transit_number: data.bank_transit_number || null,
      bank_institution_number: data.bank_institution_number || null,
      bank_account_number: data.bank_account_number || null,
      bank_iban: data.bank_iban || null,
      bank_swift_bic: data.bank_swift_bic || null,
      bank_currency: data.bank_currency || null,
    };

    if (editingContact) {
      const { error } = await supabase
        .from("contacts")
        .update({
          ...contactData,
          updated_at: new Date().toISOString(),
        })
        .eq("id", editingContact.id);

      if (error) {
        toast({
          title: "Error",
          description: "Failed to update contact",
          variant: "destructive",
        });
        throw error;
      }

      toast({ title: "Contact updated" });
    } else {
      const { error } = await supabase.from("contacts").insert({
        user_id: user.id,
        ...contactData,
      });

      if (error) {
        toast({
          title: "Error",
          description: "Failed to add contact",
          variant: "destructive",
        });
        throw error;
      }

      toast({ title: "Contact added" });
    }

    setEditingContact(null);
    fetchContacts();
  };

  const handleDelete = async () => {
    if (!contactToDelete) return;

    const { error } = await supabase
      .from("contacts")
      .delete()
      .eq("id", contactToDelete.id);

    if (error) {
      toast({
        title: "Error",
        description: "Failed to delete contact",
        variant: "destructive",
      });
    } else {
      setContacts(contacts.filter((c) => c.id !== contactToDelete.id));
      toast({ title: "Contact deleted" });
    }
    setContactToDelete(null);
  };

  const hasBankDetails = (contact: Tables<"contacts">) => {
    return !!(contact.bank_account_number || contact.bank_iban);
  };

  const filteredContacts = contacts.filter((contact) => {
    const q = searchQuery.toLowerCase();
    return (
      contact.name.toLowerCase().includes(q) ||
      (contact.email && contact.email.toLowerCase().includes(q)) ||
      (contact.role && contact.role.toLowerCase().includes(q))
    );
  });

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => navigate(-1)}
            >
              <ArrowLeft className="w-4 h-4 mr-1" /> Back
            </Button>
            <div className="w-px h-6 bg-border" />
            <div
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => navigate("/dashboard")}
            >
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center p-1.5">
                <Music className="w-full h-full text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <Button variant="outline" onClick={() => navigate("/payments")}>
              Payment History
            </Button>
            <Button variant="outline" onClick={() => navigate("/dashboard")}>
              Back to Dashboard
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-foreground mb-2">Contacts</h2>
            <p className="text-muted-foreground">
              Manage your collaborators and payees
            </p>
          </div>
          <Button
            onClick={() => {
              setEditingContact(null);
              setFormOpen(true);
            }}
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Contact
          </Button>
        </div>

        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search by name, email, or role..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground">Loading contacts...</p>
          </div>
        ) : (
          <>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredContacts.map((contact) => (
                <Card key={contact.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
                          {contact.name.charAt(0).toUpperCase()}
                        </div>
                        <div>
                          <CardTitle className="text-lg">{contact.name}</CardTitle>
                          <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                            {contact.role && (
                              <Badge variant="secondary">
                                {contact.role}
                              </Badge>
                            )}
                            {hasBankDetails(contact) ? (
                              <Badge className="text-xs bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300 hover:bg-green-100 dark:hover:bg-green-900">
                                <Building2 className="w-3 h-3 mr-1" />
                                Bank Added
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="text-xs text-muted-foreground">
                                <Building2 className="w-3 h-3 mr-1" />
                                No Bank Details
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {contact.email && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Mail className="w-3.5 h-3.5" />
                        <span className="truncate">{contact.email}</span>
                      </div>
                    )}
                    {contact.phone && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Phone className="w-3.5 h-3.5" />
                        <span>{contact.phone}</span>
                      </div>
                    )}

                    <div className="flex gap-2 pt-2">
                      {hasBankDetails(contact) && (
                        <Button
                          variant="default"
                          size="sm"
                          className="flex-1"
                          onClick={() => setPaymentContact(contact)}
                        >
                          <DollarSign className="w-3.5 h-3.5 mr-1" />
                          Pay
                        </Button>
                      )}
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1"
                        onClick={() => {
                          setEditingContact(contact);
                          setFormOpen(true);
                        }}
                      >
                        <Pencil className="w-3.5 h-3.5 mr-1" />
                        Edit
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="text-destructive hover:bg-destructive hover:text-destructive-foreground"
                        onClick={() => setContactToDelete(contact)}
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {filteredContacts.length === 0 && (
              <div className="text-center py-12">
                <p className="text-muted-foreground">
                  {searchQuery ? "No contacts match your search" : "No contacts yet"}
                </p>
              </div>
            )}
          </>
        )}
      </main>

      <ContactFormDialog
        open={formOpen}
        onOpenChange={(open) => {
          setFormOpen(open);
          if (!open) setEditingContact(null);
        }}
        contact={editingContact}
        onSave={handleSave}
      />

      <PaymentDialog
        open={!!paymentContact}
        onOpenChange={(open) => { if (!open) setPaymentContact(null); }}
        contact={paymentContact}
        onPaymentComplete={() => {
          setPaymentContact(null);
          toast({ title: "Payment processed successfully" });
        }}
      />

      <AlertDialog open={!!contactToDelete} onOpenChange={() => setContactToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete contact?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{contactToDelete?.name}</strong>. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default Contacts;
