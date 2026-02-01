import { useState } from "react";
import { FileUploadZone } from "@/components/FileUploadZone";
import { DocumentList } from "@/components/DocumentList";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { hfApi } from "@/lib/api-client";
import { useKnowledgeStore } from "@/lib/knowledge-store";
import type { Document } from "@shared/schema";

// Store File objects alongside document metadata
interface DocumentWithFile extends Document {
  file?: File; // Store the actual File object
  processingDetails?: {
    agent?: string;
    factsAdded?: number;
    rowsProcessed?: number;
    columns?: number;
    csvStats?: {
      columnFacts?: number;
      rowFacts?: number;
      relationshipFacts?: number;
      entityColumns?: string[];
    };
  };
}

export default function UploadPage() {
  const [documents, setDocuments] = useState<DocumentWithFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const { toast } = useToast();
  const { refreshFacts } = useKnowledgeStore();

  const handleFilesSelected = (files: File[]) => {
    const newDocuments: DocumentWithFile[] = files.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      type: file.name.split('.').pop() as any,
      size: file.size,
      uploadedAt: new Date().toISOString(),
      status: "pending" as const,
      file: file, // Store the actual File object
    }));

    setDocuments((prev) => [...prev, ...newDocuments]);
    toast({
      title: "Files added",
      description: `${files.length} file(s) ready for processing`,
    });
  };

  const handleRemove = (id: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== id));
  };

  const handleProcess = async () => {
    const pendingDocs = documents.filter((doc) => doc.status === "pending");
    if (pendingDocs.length === 0) {
      return;
    }

    setIsProcessing(true);
    setDocuments((prev) =>
      prev.map((doc) =>
        doc.status === "pending" ? { ...doc, status: "processing" as const } : doc
      )
    );
    
    toast({
      title: "Processing started",
      description: `Uploading and processing ${pendingDocs.length} document(s)...`,
    });

    try {
      // Get the actual File objects from pending documents
      const fileObjects: File[] = [];
      for (const doc of pendingDocs) {
        if (doc.file) {
          fileObjects.push(doc.file);
        } else {
          console.warn(`Document ${doc.name} has no file object, skipping`);
        }
      }

      if (fileObjects.length === 0) {
        throw new Error("No files to upload");
      }

      console.log(`Uploading ${fileObjects.length} file(s) to backend...`);
      const result = await hfApi.uploadDocuments(fileObjects);
      
      if (result.success) {
        // Parse result to extract CSV-specific information
        const responseData = result.data;
        const fileResults = responseData?.file_results || [];
        
        console.log('📊 Upload response:', responseData);
        console.log('📊 File results:', fileResults);
        
        setDocuments((prev) =>
          prev.map((doc) => {
            if (doc.status === "processing") {
              // Find matching result for this document
              const fileResult = fileResults.find((fr: any) => 
                fr.filename === doc.name || fr.name === doc.name
              );
              
              console.log(`📊 Processing doc ${doc.name}, found result:`, fileResult);
              
              const isCSV = doc.type === 'csv';
              const processingDetails: any = {};
              
              if (fileResult) {
                processingDetails.agent = fileResult.agent || fileResult.agent_name;
                processingDetails.factsAdded = fileResult.facts_added ?? responseData?.facts_extracted;
                
                if (isCSV && fileResult.csv_stats) {
                  processingDetails.csvStats = {
                    columnFacts: fileResult.csv_stats.column_facts,
                    rowFacts: fileResult.csv_stats.row_facts,
                    relationshipFacts: fileResult.csv_stats.relationship_facts,
                    entityColumns: fileResult.csv_stats.entity_columns || []
                  };
                  processingDetails.rowsProcessed = fileResult.rows_processed;
                  processingDetails.columns = fileResult.total_columns;
                } else if (fileResult.metadata) {
                  processingDetails.rowsProcessed = fileResult.metadata.rows;
                  processingDetails.columns = fileResult.metadata.columns;
                }
              } else if (responseData?.facts_extracted != null) {
                // No per-file result; use response-level counts
                processingDetails.factsAdded = responseData.facts_extracted;
              }
              
              return { 
                ...doc, 
                status: "completed" as const,
                processingDetails: Object.keys(processingDetails).length > 0 ? processingDetails : undefined
              };
            }
            return doc;
          })
        );
        
        // Refresh facts to show newly extracted facts
        if (refreshFacts) {
          console.log('🔄 Upload: Refreshing facts after document processing...');
          await refreshFacts();
          console.log('✅ Upload: Facts refreshed');
        } else {
          console.warn('⚠️ Upload: refreshFacts not available');
        }
        
        // Show detailed toast for CSV files
        const csvFiles = pendingDocs.filter(doc => doc.type === 'csv');
        if (csvFiles.length > 0) {
          const csvFile = csvFiles[0];
          const doc = documents.find(d => d.id === csvFile.id);
          const details = doc?.processingDetails;
          
          if (details?.csvStats) {
            toast({
              title: "CSV Processing Complete",
              description: `Processed ${details.rowsProcessed} rows, ${details.columns} columns. Added ${details.factsAdded} facts to knowledge graph.`,
            });
          } else {
            toast({
              title: "Processing complete",
              description: result.data?.message || "Knowledge extraction finished successfully",
            });
          }
        } else {
          toast({
            title: "Processing complete",
            description: result.data?.message || "Knowledge extraction finished successfully",
          });
        }
      } else {
        // Mark as failed
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.status === "processing" ? { ...doc, status: "pending" as const } : doc
          )
        );
        toast({
          title: "Processing failed",
          description: result.error || "Failed to process documents",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error processing documents:", error);
      setDocuments((prev) =>
        prev.map((doc) =>
          doc.status === "processing" ? { ...doc, status: "pending" as const } : doc
        )
      );
      toast({
        title: "Processing failed",
        description: error instanceof Error ? error.message : "Failed to process documents",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const hasPendingDocs = documents.some((doc) => doc.status === "pending");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold mb-2">Upload Documents</h1>
        <p className="text-muted-foreground">
          Upload research documents to extract knowledge and build your knowledge graph
        </p>
      </div>

      <FileUploadZone onFilesSelected={handleFilesSelected} />

      {documents.length > 0 && (
        <>
          <DocumentList documents={documents} onRemove={handleRemove} />
          <div className="flex gap-2">
            <Button
              onClick={handleProcess}
              disabled={!hasPendingDocs || isProcessing}
              data-testid="button-process-documents"
            >
              {isProcessing ? "Processing..." : "Process Documents"}
            </Button>
            <Button
              variant="outline"
              onClick={() => setDocuments([])}
              data-testid="button-clear-all"
            >
              Clear All
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
