import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Music, AlertCircle, ArrowLeft, FileText, Download, Loader2 } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useAuth } from "@/contexts/AuthContext";

// Backend API URL
const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

interface ProjectFile {
  id: string;
  project_id: string;
  folder_category: string;
  file_name: string;
  file_url: string;
  file_path: string;
  file_size: number;
  file_type: string;
  created_at: string;
}

interface RoyaltyPayment {
  song_title: string;
  party_name: string;
  role: string;
  royalty_type: string;
  percentage: number;
  total_royalty: number;
  amount_to_pay: number;
  terms?: string;
}

interface CalculationResult {
  status: string;
  total_payments: number;
  payments: RoyaltyPayment[];
  excel_file_url?: string;
  message: string;
}

const OneClickDocuments = () => {
  const navigate = useNavigate();
  const { artistId } = useParams<{ artistId: string }>();
  const { user } = useAuth();
  
  const [contracts, setContracts] = useState<ProjectFile[]>([]);
  const [royaltyStatements, setRoyaltyStatements] = useState<ProjectFile[]>([]);
  const [selectedContract, setSelectedContract] = useState<string>("");
  const [selectedStatement, setSelectedStatement] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [isCalculating, setIsCalculating] = useState(false);
  const [error, setError] = useState<string>("");
  const [calculationResult, setCalculationResult] = useState<CalculationResult | null>(null);

  // Fetch contracts and royalty statements for the artist
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        setIsLoading(true);
        
        // Fetch contracts
        const contractsRes = await fetch(`${API_URL}/files/artist/${artistId}/category/contract`);
        if (!contractsRes.ok) throw new Error("Failed to fetch contracts");
        const contractsData = await contractsRes.json();
        setContracts(contractsData);

        // Fetch royalty statements
        const statementsRes = await fetch(`${API_URL}/files/artist/${artistId}/category/royalty_statement`);
        if (!statementsRes.ok) throw new Error("Failed to fetch royalty statements");
        const statementsData = await statementsRes.json();
        setRoyaltyStatements(statementsData);

        setIsLoading(false);
      } catch (err) {
        console.error("Error fetching files:", err);
        setError("Failed to load files. Please check your backend connection.");
        setIsLoading(false);
      }
    };

    if (artistId) {
      fetchFiles();
    }
  }, [artistId]);

  const handleCalculate = async () => {
    setError("");
    setCalculationResult(null);

    if (!selectedContract) {
      setError("Please select a contract");
      return;
    }

    if (!selectedStatement) {
      setError("Please select a royalty statement");
      return;
    }

    try {
      setIsCalculating(true);

      // Get the selected contract to find its project_id
      const contract = contracts.find(c => c.id === selectedContract);
      if (!contract) {
        throw new Error("Selected contract not found");
      }

      if (!user) {
        throw new Error("User not authenticated");
      }

      const response = await fetch(`${API_URL}/oneclick/calculate-royalties`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contract_id: selectedContract,
          user_id: user.id,
          project_id: contract.project_id,
          royalty_statement_file_id: selectedStatement,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to calculate royalties");
      }

      const result: CalculationResult = await response.json();
      setCalculationResult(result);
    } catch (err: any) {
      console.error("Error calculating royalties:", err);
      setError(err.message || "Failed to calculate royalties. Please try again.");
    } finally {
      setIsCalculating(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/tools/oneclick")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Artist Selection
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">OneClick Royalty Calculator</h2>
          <p className="text-muted-foreground">Select a contract and royalty statement to calculate payments</p>
        </div>

        <div className="grid gap-6">
          {/* File Selection Card */}
          <Card>
            <CardHeader>
              <CardTitle>Select Documents</CardTitle>
              <CardDescription>Choose a contract and royalty statement to process</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {isLoading ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                  Loading documents...
                </div>
              ) : (
                <>
                  {/* Contract Selection */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-foreground">Contract</label>
                    <Select value={selectedContract} onValueChange={setSelectedContract}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a contract" />
                      </SelectTrigger>
                      <SelectContent>
                        {contracts.length === 0 ? (
                          <SelectItem value="none" disabled>No contracts available</SelectItem>
                        ) : (
                          contracts.map((contract) => (
                            <SelectItem key={contract.id} value={contract.id}>
                              <div className="flex items-center gap-2">
                                <FileText className="w-4 h-4" />
                                {contract.file_name}
                              </div>
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Royalty Statement Selection */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-foreground">Royalty Statement</label>
                    <Select value={selectedStatement} onValueChange={setSelectedStatement}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a royalty statement" />
                      </SelectTrigger>
                      <SelectContent>
                        {royaltyStatements.length === 0 ? (
                          <SelectItem value="none" disabled>No royalty statements available</SelectItem>
                        ) : (
                          royaltyStatements.map((statement) => (
                            <SelectItem key={statement.id} value={statement.id}>
                              <div className="flex items-center gap-2">
                                <FileText className="w-4 h-4" />
                                {statement.file_name}
                              </div>
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Calculate Button */}
                  <Button
                    onClick={handleCalculate}
                    className="w-full"
                    disabled={!selectedContract || !selectedStatement || isCalculating}
                  >
                    {isCalculating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Calculating...
                      </>
                    ) : (
                      "Calculate Royalties"
                    )}
                  </Button>
                </>
              )}
            </CardContent>
          </Card>

          {/* Error Alert */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Results Card */}
          {calculationResult && (
            <Card>
              <CardHeader>
                <CardTitle>Calculation Results</CardTitle>
                <CardDescription>{calculationResult.message}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Song Title</TableHead>
                        <TableHead>Payee</TableHead>
                        <TableHead>Role</TableHead>
                        <TableHead>Royalty Type</TableHead>
                        <TableHead className="text-right">Share %</TableHead>
                        <TableHead className="text-right">Total Royalty</TableHead>
                        <TableHead className="text-right">Amount to Pay</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {calculationResult.payments.map((payment, index) => (
                        <TableRow key={index}>
                          <TableCell className="font-medium">{payment.song_title}</TableCell>
                          <TableCell>{payment.party_name}</TableCell>
                          <TableCell className="capitalize">{payment.role}</TableCell>
                          <TableCell className="capitalize">{payment.royalty_type}</TableCell>
                          <TableCell className="text-right">{payment.percentage}%</TableCell>
                          <TableCell className="text-right">{formatCurrency(payment.total_royalty)}</TableCell>
                          <TableCell className="text-right font-semibold">{formatCurrency(payment.amount_to_pay)}</TableCell>
                        </TableRow>
                      ))}
                      <TableRow className="bg-muted/50 font-bold">
                        <TableCell colSpan={6} className="text-right">Total Payments:</TableCell>
                        <TableCell className="text-right">
                          {formatCurrency(calculationResult.payments.reduce((sum, p) => sum + p.amount_to_pay, 0))}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
};

export default OneClickDocuments;
