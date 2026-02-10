import { useState, useMemo } from "react";
import { Edit2, Trash2, Search, Info, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Fact } from "@shared/schema";

interface KnowledgeBaseTableProps {
  facts: Fact[];
  onEdit: (fact: Fact) => void;
  onDelete: (id: string) => void;
}

export function KnowledgeBaseTable({ facts, onEdit, onDelete }: KnowledgeBaseTableProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [showInferred, setShowInferred] = useState(true);
  const [minConfidence, setMinConfidence] = useState([0.0]);
  const [selectedSource, setSelectedSource] = useState<string>("all");
  
  // Get all unique source documents from facts
  const availableSources = useMemo(() => {
    const sources = new Set<string>();
    facts.forEach((fact) => {
      if (fact.sourceDocuments && fact.sourceDocuments.length > 0) {
        fact.sourceDocuments.forEach((source) => {
          if (source.document) {
            sources.add(source.document);
          }
        });
      } else if (fact.sourceDocument) {
        sources.add(fact.sourceDocument);
      }
    });
    return Array.from(sources).sort();
  }, [facts]);

  const filteredFacts = facts.filter(
    (fact) => {
      // Apply search filter
      const matchesSearch = 
        fact.subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
        fact.predicate.toLowerCase().includes(searchTerm.toLowerCase()) ||
        fact.object.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (fact.details && fact.details.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (fact.sourceDocument && fact.sourceDocument.toLowerCase().includes(searchTerm.toLowerCase()));
      
      if (!matchesSearch) return false;
      
      // Apply source filter
      if (selectedSource !== "all") {
        let matchesSource = false;
        // Check if fact has the selected source in sourceDocuments array
        if (fact.sourceDocuments && fact.sourceDocuments.length > 0) {
          matchesSource = fact.sourceDocuments.some(
            (source) => source.document === selectedSource
          );
        }
        // Also check the legacy sourceDocument field
        if (!matchesSource && fact.sourceDocument === selectedSource) {
          matchesSource = true;
        }
        if (!matchesSource) return false;
      }
      
      // Apply inferred filter
      if (!showInferred && (fact.isInferred || fact.type === "inferred")) {
        return false;
      }
      
      // Apply confidence filter
      const confidence = fact.confidence ?? 0.7;
      if (confidence < minConfidence[0]) {
        return false;
      }
      
      return true;
    }
  );

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between gap-4 mb-6">
        <h3 className="text-lg font-semibold">Knowledge Base Facts</h3>
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search facts..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9"
            data-testid="input-search-facts"
          />
        </div>
      </div>
      
      {/* Filters */}
      <div className="flex items-center gap-6 mb-4 p-4 bg-muted/50 rounded-lg">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Filters:</span>
        </div>
        
        <div className="flex items-center gap-2">
          <Switch
            id="show-inferred"
            checked={showInferred}
            onCheckedChange={setShowInferred}
          />
          <Label htmlFor="show-inferred" className="text-sm cursor-pointer">
            Show Inferred Facts
          </Label>
        </div>
        
        <div className="flex items-center gap-3 flex-1 max-w-xs">
          <Label className="text-sm whitespace-nowrap">
            Min Confidence: {minConfidence[0].toFixed(1)}
          </Label>
          <Slider
            value={minConfidence}
            onValueChange={setMinConfidence}
            min={0}
            max={1}
            step={0.1}
            className="flex-1"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <Label htmlFor="source-filter" className="text-sm whitespace-nowrap">
            Source:
          </Label>
          <Select value={selectedSource} onValueChange={setSelectedSource}>
            <SelectTrigger id="source-filter" className="w-[200px]">
              <SelectValue placeholder="All sources" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sources</SelectItem>
              {availableSources.map((source) => (
                <SelectItem key={source} value={source}>
                  {source.length > 30 ? `${source.substring(0, 30)}...` : source}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {filteredFacts.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <p className="text-muted-foreground mb-2">
            {searchTerm ? "No facts match your search" : "No facts in knowledge base"}
          </p>
          <p className="text-sm text-muted-foreground">
            Upload documents to extract knowledge
          </p>
        </div>
      ) : (
        <div className="border rounded-md">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[18%]">Subject</TableHead>
                <TableHead className="w-[13%]">Predicate</TableHead>
                <TableHead className="w-[18%]">Object</TableHead>
                <TableHead className="w-[10%]">Details</TableHead>
                <TableHead className="w-[12%]">Source</TableHead>
                <TableHead className="w-[12%]">Uploaded</TableHead>
                <TableHead className="w-[8%]">Type</TableHead>
                <TableHead className="w-[8%]">Confidence</TableHead>
                <TableHead className="w-[10%]">Agent</TableHead>
                <TableHead className="w-[5%] text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredFacts.map((fact) => (
                <TableRow key={fact.id} data-testid={`fact-row-${fact.id}`}>
                  <TableCell className="font-mono text-sm">{fact.subject}</TableCell>
                  <TableCell className="font-mono text-sm text-muted-foreground">
                    {fact.predicate}
                  </TableCell>
                  <TableCell className="font-mono text-sm">{fact.object}</TableCell>
                  <TableCell>
                    {fact.details ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-6 w-6">
                              <Info className="h-4 w-4 text-muted-foreground" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent className="max-w-md">
                            <p className="text-sm whitespace-pre-wrap">{fact.details}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <span className="text-muted-foreground text-xs">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm">
                    {fact.sourceDocuments && fact.sourceDocuments.length > 0 ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="text-muted-foreground cursor-help">
                              {fact.sourceDocuments.length === 1 
                                ? (fact.sourceDocuments[0].document.length > 20 
                                    ? `${fact.sourceDocuments[0].document.substring(0, 20)}...` 
                                    : fact.sourceDocuments[0].document)
                                : `${fact.sourceDocuments.length} sources`}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent className="max-w-md">
                            <div className="space-y-1">
                              {fact.sourceDocuments.map((source, idx) => (
                                <div key={idx} className="text-sm">
                                  <p className="font-medium">{source.document}</p>
                                  {source.uploadedAt && (
                                    <p className="text-xs text-muted-foreground">
                                      {new Date(source.uploadedAt).toLocaleString()}
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : fact.sourceDocument ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="text-muted-foreground cursor-help">
                              {fact.sourceDocument.length > 20 
                                ? `${fact.sourceDocument.substring(0, 20)}...` 
                                : fact.sourceDocument}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-sm">{fact.sourceDocument}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <span className="text-muted-foreground text-xs">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {fact.uploadedAt ? (
                      new Date(fact.uploadedAt).toLocaleDateString(undefined, {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })
                    ) : (
                      <span className="text-muted-foreground text-xs">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm">
                    {(fact.type === "inferred" || fact.isInferred) ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="text-blue-400 font-medium cursor-help">Inferred</span>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-sm">This fact was inferred through transitive reasoning</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <span className="text-muted-foreground text-xs">Original</span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm">
                    <span className="text-muted-foreground">
                      {fact.confidence !== undefined && fact.confidence !== null 
                        ? fact.confidence.toFixed(2) 
                        : "0.70"}
                    </span>
                  </TableCell>
                  <TableCell className="text-sm">
                    {fact.agent ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 cursor-help">
                              {fact.agent.replace(' Agent', '')}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-sm">Processed by {fact.agent}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <span className="text-muted-foreground text-xs">Manual</span>
                    )}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => onEdit(fact)}
                        data-testid={`button-edit-${fact.id}`}
                      >
                        <Edit2 className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => onDelete(fact.id)}
                        data-testid={`button-delete-${fact.id}`}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
      
      <div className="flex items-center justify-between mt-4 text-sm text-muted-foreground">
        <p>
          Showing {filteredFacts.length} of {facts.length} facts
        </p>
      </div>
    </Card>
  );
}
