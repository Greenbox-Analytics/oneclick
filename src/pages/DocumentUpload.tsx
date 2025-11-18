// React hooks for managing component state
import { useState } from "react";
// UI components from shadcn/ui library for building the interface
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";
// Icons from lucide-react for visual elements
import { Music, ArrowLeft, Upload, FileText, X, FileSignature, Receipt, Users, DollarSign, Download, FileSpreadsheet } from "lucide-react";
// React Router hooks for navigation and getting URL parameters
import { useNavigate, useParams } from "react-router-dom";
// Recharts for pie chart
import { PieChart, Pie, Cell, Legend } from "recharts";
// xlsx library for Excel export
import * as XLSX from "xlsx";

// Type definitions for royalty calculation results
interface RoyaltyBreakdown {
  songName: string;
  contributorName: string;
  role: string;
  royaltyPercentage: number;
  amount: number;
}

interface RoyaltyResults {
  songTitle: string;
  totalContributors: number;
  totalRevenue: number;
  breakdown: RoyaltyBreakdown[];
}

/**
 * DocumentUpload Component
 * 
 * This component allows users to upload documents (contracts, agreements, etc.)
 * for a specific artist. The artist ID is passed via the URL route parameter.
 * 
 * Route: /oneclick/:artistId/documents
 */
const DocumentUpload = () => {
  // Hook to programmatically navigate to different routes
  const navigate = useNavigate();
  
  // Extract the artistId from the URL parameters (e.g., from /oneclick/1/documents)
  // The artistId comes from the route parameter defined in App.tsx
  const { artistId } = useParams<{ artistId: string }>();
  
  // State to store contract files separately
  // File[] is an array of File objects (native browser File API)
  const [contractFiles, setContractFiles] = useState<File[]>([]);
  
  // State to store royalty statement files separately
  const [royaltyStatementFiles, setRoyaltyStatementFiles] = useState<File[]>([]);
  
  // State to track whether files are currently being uploaded to the server
  // Used to show loading state and disable buttons during upload
  const [isUploading, setIsUploading] = useState(false);
  
  // State to store royalty calculation results
  // null means no results yet, object contains the calculated royalty data
  const [royaltyResults, setRoyaltyResults] = useState<RoyaltyResults | null>(null);

  // Temporary mock data for artists - TODO: Replace with API call to fetch real artist data
  // This should eventually fetch from your backend using the artistId
  const mockArtists = [
    { id: 1, name: "Luna Rivers", hasContract: true },
    { id: 2, name: "The Echoes", hasContract: true },
    { id: 3, name: "DJ Neon", hasContract: false },
  ];

  // Find the artist object that matches the artistId from the URL
  // Converts artistId (string) to number for comparison
  // Returns undefined if no artist is found
  const artist = mockArtists.find(a => a.id === Number(artistId));

  /**
   * Handles contract file selection from the file input
   * 
   * When user selects contract files (via the file input), this function:
   * 1. Gets the selected files from the input element
   * 2. Converts the FileList to a regular array
   * 3. Adds the new files to the existing contractFiles state
   * 
   * @param e - The change event from the file input element
   */
  const handleContractFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Get the files from the input element (FileList object)
    const files = e.target.files;
    if (files) {
      // Convert FileList to array (FileList is array-like but not a real array)
      const fileArray = Array.from(files);
      // Add new files to existing contract files using spread operator
      setContractFiles(prev => [...prev, ...fileArray]);
    }
  };

  /**
   * Handles royalty statement file selection from the file input
   * 
   * When user selects royalty statement files (via the file input), this function:
   * 1. Gets the selected files from the input element
   * 2. Converts the FileList to a regular array
   * 3. Adds the new files to the existing royaltyStatementFiles state
   * 
   * @param e - The change event from the file input element
   */
  const handleRoyaltyStatementFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Get the files from the input element (FileList object)
    const files = e.target.files;
    if (files) {
      // Convert FileList to array (FileList is array-like but not a real array)
      const fileArray = Array.from(files);
      // Add new files to existing royalty statement files using spread operator
      setRoyaltyStatementFiles(prev => [...prev, ...fileArray]);
    }
  };

  /**
   * Removes a contract file from the upload list
   * 
   * When user clicks the X button on a contract file, this removes it from the list
   * before uploading. The file is identified by its index in the array.
   * 
   * @param index - The index of the file to remove from contractFiles array
   */
  const handleRemoveContractFile = (index: number) => {
    // Filter out the file at the specified index
    setContractFiles(prev => prev.filter((_, i) => i !== index));
  };

  /**
   * Removes a royalty statement file from the upload list
   * 
   * When user clicks the X button on a royalty statement file, this removes it from the list
   * before uploading. The file is identified by its index in the array.
   * 
   * @param index - The index of the file to remove from royaltyStatementFiles array
   */
  const handleRemoveRoyaltyStatementFile = (index: number) => {
    // Filter out the file at the specified index
    setRoyaltyStatementFiles(prev => prev.filter((_, i) => i !== index));
  };

  /**
   * Handles the calculation of royalties after documents are uploaded
   * 
   * This function:
   * 1. Checks if there are files to upload (contracts and royalty statements)
   * 2. Sets loading state to true (disables buttons, shows "Calculating...")
   * 3. Processes files and calculates royalties (currently mocked with dummy data)
   * 4. Displays results on the same page
   * 
   * TODO: Replace with actual API call to process documents and calculate royalties
   */
  const handleCalculateRoyalties = async () => {
    // Don't proceed if both file types are not uploaded
    if (contractFiles.length === 0 || royaltyStatementFiles.length === 0) return;

    // Set uploading state to true - this will:
    // - Change button text to "Calculating..."
    // - Disable the upload and cancel buttons
    setIsUploading(true);
    
    // TODO: Add backend logic to upload and process documents
    // This should make API calls to:
    // 1. Upload contract files: await uploadContractFiles(contractFiles, artistId);
    // 2. Upload royalty statement files: await uploadRoyaltyStatementFiles(royaltyStatementFiles, artistId);
    // 3. Process documents and calculate royalties: await calculateRoyalties(artistId);
    
    // Mock calculation delay (simulates network request and processing)
    // In real implementation, replace this with actual API call
    setTimeout(() => {
      setIsUploading(false);
      
      // Dummy data for royalty calculation results
      // TODO: Replace with actual API response
      const dummyResults: RoyaltyResults = {
        songTitle: "Midnight Dreams",
        totalContributors: 4,
        totalRevenue: 125000.00,
        breakdown: [
          {
            songName: "Midnight Dreams",
            contributorName: "Luna Rivers",
            role: "Artist",
            royaltyPercentage: 45.0,
            amount: 56250.00
          },
          {
            songName: "Midnight Dreams",
            contributorName: "Alex Martinez",
            role: "Producer",
            royaltyPercentage: 30.0,
            amount: 37500.00
          },
          {
            songName: "Midnight Dreams",
            contributorName: "Sarah Chen",
            role: "Songwriter",
            royaltyPercentage: 20.0,
            amount: 25000.00
          },
          {
            songName: "Midnight Dreams",
            contributorName: "Mike Johnson",
            role: "Featured Artist",
            royaltyPercentage: 5.0,
            amount: 6250.00
          }
        ]
      };
      
      setRoyaltyResults(dummyResults);
    }, 2000);
  };

  /**
   * Exports the royalty breakdown table to CSV format
   * 
   * Creates a CSV file with all the royalty breakdown data and triggers download
   */
  const handleExportCSV = () => {
    if (!royaltyResults) return;

    // Create CSV header
    const headers = ["Song Name", "Contributor Name", "Role", "Royalty Share %", "Amount"];
    
    // Create CSV rows
    const rows = royaltyResults.breakdown.map(row => [
      row.songName,
      row.contributorName,
      row.role,
      `${row.royaltyPercentage}%`,
      `$${row.amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    ]);

    // Combine headers and rows
    const csvContent = [
      headers.join(","),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(","))
    ].join("\n");

    // Create blob and download
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `royalty-breakdown-${royaltyResults.songTitle.replace(/\s+/g, "-")}.csv`);
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  /**
   * Exports the royalty breakdown table to Excel format
   * 
   * Creates an Excel file with all the royalty breakdown data and triggers download
   */
  const handleExportExcel = () => {
    if (!royaltyResults) return;

    // Prepare data for Excel
    const excelData = [
      ["Song Name", "Contributor Name", "Role", "Royalty Share %", "Amount"],
      ...royaltyResults.breakdown.map(row => [
        row.songName,
        row.contributorName,
        row.role,
        row.royaltyPercentage,
        row.amount
      ])
    ];

    // Create workbook and worksheet
    const ws = XLSX.utils.aoa_to_sheet(excelData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Royalty Breakdown");

    // Generate Excel file and download
    XLSX.writeFile(wb, `royalty-breakdown-${royaltyResults.songTitle.replace(/\s+/g, "-")}.xlsx`);
  };

  return (
    // Main container - min-h-screen ensures it takes full viewport height
    <div className="min-h-screen bg-background">
      {/* Header section with logo and navigation */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          {/* Logo and app name */}
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii AI</h1>
          </div>
          {/* Back button - navigates to the tools page */}
          <Button variant="outline" onClick={() => navigate("/tools")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Tools
          </Button>
        </div>
      </header>

      {/* Main content area */}
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Page title section */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">Upload Documents</h2>
          {/* Display artist name if found, otherwise show generic text */}
          <p className="text-muted-foreground">
            Upload documents for {artist?.name || "the selected artist"}
          </p>
        </div>

        {/* Two side-by-side cards for Contract and Royalty Statement uploads */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Contract Upload Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileSignature className="w-5 h-5 text-primary" />
                Upload Contract
              </CardTitle>
              <CardDescription>
                Upload artist contract documents
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Contract file upload drop zone area */}
              <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                <FileSignature className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                <p className="text-foreground font-medium mb-2 text-sm">Upload Contract</p>
                <p className="text-muted-foreground mb-4 text-xs">
                  PDF or Word DOCX files only
                </p>
                {/* Hidden file input for contracts - accepts only PDF and DOCX */}
                <Input
                  id="contract-upload"
                  type="file"
                  multiple
                  accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  onChange={handleContractFileChange}
                  className="hidden"
                />
                <label htmlFor="contract-upload">
                  <Button variant="outline" size="sm" asChild>
                    <span>
                      <Upload className="w-4 h-4 mr-2" />
                      Select Files
                    </span>
                  </Button>
                </label>
              </div>

              {/* Contract files list */}
              {contractFiles.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-xs font-medium text-foreground">Contract Files:</h3>
                  <div className="space-y-2">
                    {contractFiles.map((file, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-2 border border-border rounded-lg bg-secondary/50"
                      >
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <FileText className="w-4 h-4 text-primary flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-foreground truncate">{file.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {(file.size / 1024).toFixed(2)} KB
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemoveContractFile(index)}
                          className="text-destructive hover:text-destructive flex-shrink-0"
                        >
                          <X className="w-3 h-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Royalty Statement Upload Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Receipt className="w-5 h-5 text-primary" />
                Upload Royalty Statement
              </CardTitle>
              <CardDescription>
                Upload royalty statement documents
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Royalty statement file upload drop zone area */}
              <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                <Receipt className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                <p className="text-foreground font-medium mb-2 text-sm">Upload Royalty Statement</p>
                <p className="text-muted-foreground mb-4 text-xs">
                  Excel (XLSX, XLS) or CSV files only
                </p>
                {/* Hidden file input for royalty statements - accepts only Excel and CSV */}
                <Input
                  id="royalty-upload"
                  type="file"
                  multiple
                  accept=".xlsx,.xls,.csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,text/csv"
                  onChange={handleRoyaltyStatementFileChange}
                  className="hidden"
                />
                <label htmlFor="royalty-upload">
                  <Button variant="outline" size="sm" asChild>
                    <span>
                      <Upload className="w-4 h-4 mr-2" />
                      Select Files
                    </span>
                  </Button>
                </label>
              </div>

              {/* Royalty statement files list */}
              {royaltyStatementFiles.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-xs font-medium text-foreground">Royalty Statement Files:</h3>
                  <div className="space-y-2">
                    {royaltyStatementFiles.map((file, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-2 border border-border rounded-lg bg-secondary/50"
                      >
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <FileText className="w-4 h-4 text-primary flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-foreground truncate">{file.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {(file.size / 1024).toFixed(2)} KB
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemoveRoyaltyStatementFile(index)}
                          className="text-destructive hover:text-destructive flex-shrink-0"
                        >
                          <X className="w-3 h-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Action buttons at the bottom - centered */}
        <div className="flex gap-3 justify-center">
          {/* Calculate Royalties button - triggers the calculation process */}
          {/* Disabled if BOTH contract and royalty statement files are not uploaded, or currently uploading */}
          <Button
            onClick={handleCalculateRoyalties}
            disabled={(contractFiles.length === 0 || royaltyStatementFiles.length === 0) || isUploading}
          >
            {/* Conditional text: show "Calculating..." during calculation, otherwise "Calculate Royalties" */}
            {isUploading ? "Calculating..." : "Calculate Royalties"}
          </Button>
          {/* Cancel button - goes back to tools page */}
          {/* Outlined style, disabled during upload to prevent navigation interruption */}
          <Button
            variant="outline"
            onClick={() => navigate("/tools")}
            disabled={isUploading}
          >
            Cancel
          </Button>
        </div>

        {/* Royalty Calculation Results Section */}
        {/* This section expands below the upload cards when results are available */}
        {royaltyResults && (
          <div className="mt-8 space-y-6">
            {/* Results Header */}
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Royalty Calculation Results</h2>
              <p className="text-muted-foreground">Breakdown of royalty distribution for the processed documents</p>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Song Title KPI */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Song Title</CardTitle>
                  <Music className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-foreground">{royaltyResults.songTitle}</div>
                </CardContent>
              </Card>

              {/* Total Contributors KPI */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Contributors</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-foreground">{royaltyResults.totalContributors}</div>
                  <p className="text-xs text-muted-foreground">Contributors on this song</p>
                </CardContent>
              </Card>

              {/* Total Revenue KPI */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-foreground">
                    ${royaltyResults.totalRevenue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                  <p className="text-xs text-muted-foreground">Total revenue generated</p>
                </CardContent>
              </Card>
            </div>

            {/* Pie Chart - Royalty Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Royalty Distribution</CardTitle>
                <CardDescription>
                  Visual breakdown of royalty shares by contributor
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={{
                    contributor: {
                      label: "Contributor",
                    },
                  }}
                  className="h-[400px]"
                >
                  <PieChart>
                    <Pie
                      data={[...royaltyResults.breakdown]
                        .sort((a, b) => b.royaltyPercentage - a.royaltyPercentage)
                        .map((row, index, sortedArray) => {
                          // Use theme primary color (hue 150) with varying shades
                          // Biggest contributor gets darkest shade (lowest lightness), smallest gets lightest
                          const totalContributors = sortedArray.length;
                          // Lightness ranges from 25% (darkest/strongest) to 65% (lightest)
                          // Biggest contributor (index 0) gets 25%, smallest gets 65%
                          const lightness = 25 + (index / (totalContributors - 1 || 1)) * 40;
                          // Saturation ranges from 50% to 60% for richer colors
                          const saturation = 50 + (index / (totalContributors - 1 || 1)) * 10;
                          
                          return {
                            name: row.contributorName,
                            value: row.royaltyPercentage,
                            amount: row.amount,
                            fill: `hsl(150, ${saturation}%, ${lightness}%)`,
                          };
                        })}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}%`}
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {[...royaltyResults.breakdown]
                        .sort((a, b) => b.royaltyPercentage - a.royaltyPercentage)
                        .map((_, index, sortedArray) => {
                          const totalContributors = sortedArray.length;
                          const lightness = 25 + (index / (totalContributors - 1 || 1)) * 40;
                          const saturation = 50 + (index / (totalContributors - 1 || 1)) * 10;
                          return (
                            <Cell
                              key={`cell-${index}`}
                              fill={`hsl(150, ${saturation}%, ${lightness}%)`}
                            />
                          );
                        })}
                    </Pie>
                    <ChartTooltip 
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="rounded-lg border bg-background p-2 shadow-sm">
                              <div className="grid gap-2">
                                <div className="flex items-center gap-2">
                                  <div
                                    className="h-3 w-3 rounded-full"
                                    style={{ backgroundColor: data.fill }}
                                  />
                                  <span className="font-medium">{data.name}</span>
                                </div>
                                <div className="text-sm">
                                  <div className="text-muted-foreground">Royalty Share</div>
                                  <div className="font-semibold">{data.value}%</div>
                                </div>
                                <div className="text-sm">
                                  <div className="text-muted-foreground">Amount</div>
                                  <div className="font-semibold">
                                    ${data.amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Legend />
                  </PieChart>
                </ChartContainer>
              </CardContent>
            </Card>

            {/* Royalty Breakdown Table */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Royalty Breakdown</CardTitle>
                    <CardDescription>
                      Detailed breakdown of royalty distribution per contributor
                    </CardDescription>
                  </div>
                  {/* Export Button with Dropdown */}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Export
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={handleExportCSV}>
                        <FileText className="h-4 w-4 mr-2" />
                        Export as CSV
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={handleExportExcel}>
                        <FileSpreadsheet className="h-4 w-4 mr-2" />
                        Export as Excel
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Song Name</TableHead>
                      <TableHead>Contributor Name</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead className="text-right">Royalty Share %</TableHead>
                      <TableHead className="text-right">Amount</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {royaltyResults.breakdown.map((row: RoyaltyBreakdown, index: number) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{row.songName}</TableCell>
                        <TableCell>{row.contributorName}</TableCell>
                        <TableCell>{row.role}</TableCell>
                        <TableCell className="text-right">{row.royaltyPercentage}%</TableCell>
                        <TableCell className="text-right font-medium">
                          ${row.amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
};

// Export the component so it can be imported and used in other files (like App.tsx)
export default DocumentUpload;

