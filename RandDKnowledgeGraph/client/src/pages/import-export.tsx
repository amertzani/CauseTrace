import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { ImportExportPanel } from "@/components/ImportExportPanel";
import { hfApi } from "@/lib/api-client";
import { useToast } from "@/hooks/use-toast";

export default function ImportExportPage() {
  const { toast } = useToast();
  const [location] = useLocation();
  const [metadata, setMetadata] = useState({
    version: "1.2.3",
    totalFacts: 0,
    lastUpdated: new Date().toISOString(),
  });

  // Fetch real metadata from backend on mount and when navigating to this page
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        console.log("📊 Import/Export: Fetching metadata from backend...");
        // Get facts count from backend
        const factsResult = await hfApi.getKnowledgeBase();
        console.log("📊 Import/Export: Facts result:", factsResult);
        
        if (factsResult.success && factsResult.data?.facts) {
          const facts = Array.isArray(factsResult.data.facts) 
            ? factsResult.data.facts 
            : factsResult.data.facts || [];
          
          console.log(`📊 Import/Export: Setting metadata with ${facts.length} facts`);
          setMetadata({
            version: "1.2.3",
            totalFacts: facts.length,
            lastUpdated: new Date().toISOString(),
          });
        } else {
          console.warn("📊 Import/Export: Failed to fetch facts, using default metadata");
        }
      } catch (error) {
        console.error("📊 Import/Export: Error fetching metadata:", error);
      }
    };
    
    // Fetch metadata when component mounts or when navigating to this page
    if (location === "/import-export") {
      fetchMetadata();
    }
  }, [location]);

  const handleExport = async (includeInferred: boolean, minConfidence: number) => {
    try {
      toast({
        title: "Exporting knowledge base...",
        description: "Please wait while we prepare your export file.",
      });

      // Call backend API to get all facts with filters
      const result = await hfApi.exportKnowledgeBase(includeInferred, minConfidence);
      
      if (result.success && result.data) {
        const exportData = result.data;
        
        // Create blob with the actual data
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `knowledge-base-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Update metadata with actual exported data
        const exportedFacts = exportData.facts?.length || 0;
        setMetadata({
          version: exportData.metadata?.version || "1.2.3",
          totalFacts: exportedFacts,
          lastUpdated: exportData.metadata?.lastUpdated || new Date().toISOString(),
        });

        toast({
          title: "Export successful",
          description: `Exported ${exportedFacts} facts to JSON file.`,
        });
      } else {
        throw new Error(result.error || "Failed to export knowledge base");
      }
    } catch (error) {
      console.error("Export error:", error);
      toast({
        title: "Export failed",
        description: error instanceof Error ? error.message : "Failed to export knowledge base",
        variant: "destructive",
      });
    }
  };

  const handleExportCausalGraph = async () => {
    try {
      toast({
        title: "Exporting causal graph...",
        description: "Preparing KB + all datasets.",
      });
      const result = await hfApi.exportCausalGraph("all", undefined, true);
      if (result.success && result.data) {
        const blob = new Blob([JSON.stringify(result.data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `causal-graph-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        const meta = result.data.metadata || {};
        const count = meta.sources?.length ?? 0;
        toast({
          title: "Causal graph exported",
          description: `Exported ${count} source(s): KB + datasets.`,
        });
      } else {
        throw new Error(result.error || "Failed to export causal graph");
      }
    } catch (error) {
      toast({
        title: "Export failed",
        description: error instanceof Error ? error.message : "Failed to export causal graph",
        variant: "destructive",
      });
    }
  };

  const handleImport = async (file: File) => {
    try {
      toast({
        title: "Importing knowledge base...",
        description: "Please wait while we process your file.",
      });

      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          console.log("Imported data:", data);

          // Call backend import endpoint
          const result = await hfApi.importKnowledgeBase(data);
          
          if (result.success) {
            // Refresh metadata after import
            const factsResult = await hfApi.getKnowledgeBase();
            if (factsResult.success && factsResult.data?.facts) {
              const facts = Array.isArray(factsResult.data.facts) 
                ? factsResult.data.facts 
                : factsResult.data.facts || [];
              
              setMetadata({
                version: "1.2.3",
                totalFacts: facts.length,
                lastUpdated: new Date().toISOString(),
              });
            }

            toast({
              title: "Import successful",
              description: `Imported ${data.metadata?.totalFacts || data.facts?.length || 0} facts from JSON file.`,
            });
          } else {
            throw new Error(result.error || "Failed to import knowledge base");
          }
        } catch (error) {
          console.error("Import error:", error);
          toast({
            title: "Import failed",
            description: error instanceof Error ? error.message : "Invalid JSON file format or import error",
            variant: "destructive",
          });
        }
      };
      reader.readAsText(file);
    } catch (error) {
      console.error("Import file error:", error);
      toast({
        title: "Import failed",
        description: "Failed to read file",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold mb-2">Import / Export</h1>
        <p className="text-muted-foreground">
          Backup and restore your knowledge base data
        </p>
      </div>

      <ImportExportPanel
        onExport={handleExport}
        onExportCausalGraph={handleExportCausalGraph}
        onImport={handleImport}
        metadata={metadata}
      />
    </div>
  );
}
