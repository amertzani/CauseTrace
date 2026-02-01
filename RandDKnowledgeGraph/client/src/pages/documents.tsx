import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { FileText, Calendar, HardDrive } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { hfApi, getUploadedDocNamesSession } from "@/lib/api-client";
import type { Document } from "@shared/schema";

/** Hide test/demo documents from the list (e.g. test_upload.csv, upload_test.csv). */
function isTestDocument(name: string): boolean {
  if (!name) return false;
  const n = name.toLowerCase();
  return ["test_upload", "upload_test", "test_dataset", "test.csv"].some((p) => n.includes(p));
}

export default function DocumentsPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [location] = useLocation();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Load documents from backend
  useEffect(() => {
    if (location === '/documents') {
      console.log('DocumentsPage: Loading documents from backend...');
      loadDocuments();
    }
  }, [location]);

  const loadDocuments = async () => {
    setIsLoading(true);
    try {
      const result = await hfApi.getDocuments();
      if (result.success && result.data?.documents) {
        // Backend filters test data; frontend also hides test/demo documents
        const sessionUploaded = getUploadedDocNamesSession();
        const mappedDocs: Document[] = result.data.documents
          .filter((doc: any) => !isTestDocument(doc.name || ""))
          .filter((doc: any) => sessionUploaded.includes(doc.name || "")) // Only show docs uploaded this session
          .map((doc: any) => ({
            id: doc.id || String(doc.name),
            name: doc.name,
            type: doc.type || doc.name.split('.').pop() || 'unknown',
            size: doc.size || 0,
            uploadedAt: doc.uploaded_at || doc.uploadedAt || new Date().toISOString(),
            status: doc.status || 'completed',
            factsExtracted: doc.facts_extracted || 0, // Add facts extracted count
          }));
        console.log(`DocumentsPage: Loaded ${mappedDocs.length} documents with facts (filtered)`);
        setDocuments(mappedDocs);
      } else {
        console.error('DocumentsPage: Failed to load documents:', result);
        setDocuments([]);
      }
    } catch (error) {
      console.error('DocumentsPage: Error loading documents:', error);
      setDocuments([]);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredDocuments = documents.filter((doc) =>
    doc.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold mb-2">Documents</h1>
        <p className="text-muted-foreground">
          View and manage all uploaded research documents
        </p>
      </div>

      <div className="flex items-center gap-4">
        <Input
          placeholder="Search documents..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="max-w-md"
          data-testid="input-search-documents"
        />
      </div>

      {isLoading ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">Loading documents...</p>
        </div>
      ) : filteredDocuments.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">
            {searchTerm ? "No documents match your search." : "No documents uploaded yet."}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {filteredDocuments.map((doc) => (
          <Card key={doc.id} className="p-6 hover-elevate" data-testid={`doc-card-${doc.id}`}>
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-md bg-primary/10">
                <FileText className="h-6 w-6 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-medium mb-1 truncate">{doc.name}</h3>
                <div className="flex flex-wrap gap-2 mb-3">
                  <Badge variant="secondary" className="text-xs">
                    {doc.type.toUpperCase()}
                  </Badge>
                  <Badge variant="secondary" className="text-xs">
                    {doc.status}
                  </Badge>
                  {(doc as any).factsExtracted > 0 && (
                    <Badge variant="default" className="text-xs">
                      {(doc as any).factsExtracted} facts
                    </Badge>
                  )}
                </div>
                <div className="flex flex-col gap-1 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <HardDrive className="h-3 w-3" />
                    <span className="font-mono text-xs">{formatFileSize(doc.size)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="h-3 w-3" />
                    <span className="text-xs">{formatDate(doc.uploadedAt)}</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>
          ))}
        </div>
      )}
    </div>
  );
}
