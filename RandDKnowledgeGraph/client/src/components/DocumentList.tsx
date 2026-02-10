import { FileText, X, Loader2, CheckCircle2, AlertCircle, Database, FileSpreadsheet } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import type { Document } from "@shared/schema";

interface ProcessingDetails {
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
}

interface DocumentWithDetails extends Document {
  processingDetails?: ProcessingDetails;
}

interface DocumentListProps {
  documents: DocumentWithDetails[];
  onRemove: (id: string) => void;
}

const statusIcons = {
  pending: Loader2,
  processing: Loader2,
  completed: CheckCircle2,
  error: AlertCircle,
};

const statusColors = {
  pending: "text-muted-foreground",
  processing: "text-primary",
  completed: "text-green-600",
  error: "text-destructive",
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

export function DocumentList({ documents, onRemove }: DocumentListProps) {
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set());
  
  if (documents.length === 0) {
    return null;
  }

  const toggleExpanded = (docId: string) => {
    setExpandedDocs(prev => {
      const next = new Set(prev);
      if (next.has(docId)) {
        next.delete(docId);
      } else {
        next.add(docId);
      }
      return next;
    });
  };

  const getFileIcon = (type: string) => {
    if (type === 'csv') return FileSpreadsheet;
    return FileText;
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Uploaded Documents</h3>
      <div className="space-y-3">
        {documents.map((doc) => {
          const Icon = statusIcons[doc.status];
          const FileIcon = getFileIcon(doc.type);
          const isProcessing = doc.status === "processing";
          const isCompleted = doc.status === "completed";
          const isCSV = doc.type === 'csv';
          const hasDetails = doc.processingDetails && Object.keys(doc.processingDetails).length > 0;
          const isExpanded = expandedDocs.has(doc.id);
          
          return (
            <div
              key={doc.id}
              className="rounded-md border p-3"
              data-testid={`document-${doc.id}`}
            >
              <div className="flex items-center gap-4">
                <FileIcon className={`h-5 w-5 flex-shrink-0 ${
                  isCSV ? "text-blue-500" : "text-muted-foreground"
                }`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{doc.name}</p>
                      {isCSV && isCompleted && (
                        <Badge variant="outline" className="text-xs">
                          CSV
                        </Badge>
                      )}
                      {hasDetails && doc.processingDetails?.agent && (
                        <Badge variant="secondary" className="text-xs">
                          {doc.processingDetails.agent}
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className="text-xs text-muted-foreground font-mono">
                        {formatFileSize(doc.size)}
                      </span>
                      <Icon
                        className={`h-4 w-4 ${statusColors[doc.status]} ${
                          isProcessing ? "animate-spin" : ""
                        }`}
                      />
                    </div>
                  </div>
                  {isProcessing && (
                    <Progress value={65} className="h-1" />
                  )}
                  
                  {/* CSV Processing Details */}
                  {isCompleted && hasDetails && isCSV && doc.processingDetails?.csvStats && (
                    <Collapsible open={isExpanded} onOpenChange={() => toggleExpanded(doc.id)}>
                      <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground mt-2">
                        {isExpanded ? (
                          <ChevronDown className="h-3 w-3" />
                        ) : (
                          <ChevronRight className="h-3 w-3" />
                        )}
                        <span>View CSV processing details</span>
                      </CollapsibleTrigger>
                      <CollapsibleContent className="mt-2 space-y-2">
                        <div className="text-xs space-y-1 pl-4 border-l-2 border-primary/20">
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <span className="font-medium">Rows Processed:</span>{" "}
                              {doc.processingDetails.rowsProcessed || 'N/A'}
                            </div>
                            <div>
                              <span className="font-medium">Columns:</span>{" "}
                              {doc.processingDetails.columns || 'N/A'}
                            </div>
                            <div>
                              <span className="font-medium">Facts Added:</span>{" "}
                              <span className="text-green-600 font-semibold">
                                {doc.processingDetails.factsAdded || 0}
                              </span>
                            </div>
                            {doc.processingDetails.csvStats.columnFacts !== undefined && (
                              <div>
                                <span className="font-medium">Column Facts:</span>{" "}
                                {doc.processingDetails.csvStats.columnFacts}
                              </div>
                            )}
                            {doc.processingDetails.csvStats.rowFacts !== undefined && (
                              <div>
                                <span className="font-medium">Row Facts:</span>{" "}
                                {doc.processingDetails.csvStats.rowFacts}
                              </div>
                            )}
                            {doc.processingDetails.csvStats.relationshipFacts !== undefined && (
                              <div>
                                <span className="font-medium">Relationships:</span>{" "}
                                {doc.processingDetails.csvStats.relationshipFacts}
                              </div>
                            )}
                          </div>
                          {doc.processingDetails.csvStats.entityColumns && 
                           doc.processingDetails.csvStats.entityColumns.length > 0 && (
                            <div className="mt-2">
                              <span className="font-medium">Entity Columns:</span>{" "}
                              <div className="flex flex-wrap gap-1 mt-1">
                                {doc.processingDetails.csvStats.entityColumns.map((col, idx) => (
                                  <Badge key={idx} variant="outline" className="text-xs">
                                    {col}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                  
                  {/* General processing details for non-CSV files */}
                  {isCompleted && hasDetails && !isCSV && (
                    <div className="text-xs text-muted-foreground mt-2 pl-4">
                      {doc.processingDetails.factsAdded !== undefined && (
                        <span>
                          Facts added: <span className="font-semibold text-foreground">
                            {doc.processingDetails.factsAdded}
                          </span>
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => onRemove(doc.id)}
                  data-testid={`button-remove-${doc.id}`}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
