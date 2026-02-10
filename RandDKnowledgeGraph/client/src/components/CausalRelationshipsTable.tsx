import { useState, useMemo, useEffect } from "react";
import { Search, Filter, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
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
import { Badge } from "@/components/ui/badge";

export interface CausalEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  details?: string;
  confidence?: number;
  is_inferred?: boolean;
  source_document?: string;
}

interface CausalNode {
  id: string;
  label: string;
}

interface CausalRelationshipsTableProps {
  edges: CausalEdge[];
  nodes?: CausalNode[];
  sourceLabel?: string;
  /** Only show these document names in the Source filter (uploaded docs). Excludes "unknown" and invalid values. */
  allowedSourceDocuments?: string[];
}

function nodeLabel(id: string, nodes?: CausalNode[]): string {
  if (nodes) {
    const n = nodes.find((x) => x.id === id);
    if (n) return n.label;
  }
  return id.replace(/^node_/i, "").replace(/_/g, " ");
}

const INVALID_SOURCE_NAMES = new Set(["unknown", "null", "undefined", ""]);

export function CausalRelationshipsTable({ edges, nodes, sourceLabel, allowedSourceDocuments }: CausalRelationshipsTableProps) {

  const [searchTerm, setSearchTerm] = useState("");
  const [showInferred, setShowInferred] = useState(true);
  const [minConfidence, setMinConfidence] = useState(0);

  const availableSources = useMemo(() => {
    const fromEdges = new Set<string>();
    edges.forEach((e) => {
      const doc = e.source_document?.trim();
      if (!doc || INVALID_SOURCE_NAMES.has(doc.toLowerCase())) return;
      if (allowedSourceDocuments && allowedSourceDocuments.length > 0) {
        if (allowedSourceDocuments.includes(doc)) fromEdges.add(doc);
      } else {
        fromEdges.add(doc);
      }
    });
    return Array.from(fromEdges).sort();
  }, [edges, allowedSourceDocuments]);

  const [selectedSource, setSelectedSource] = useState<string>("all");

  // If current selection is no longer in the list (e.g. was "unknown"), reset to "all"
  useEffect(() => {
    if (
      selectedSource !== "all" &&
      availableSources.length > 0 &&
      !availableSources.includes(selectedSource)
    ) {
      setSelectedSource("all");
    }
  }, [availableSources, selectedSource]);

  const effectiveSource =
    selectedSource === "all" || availableSources.includes(selectedSource)
      ? selectedSource
      : "all";

  const filteredEdges = useMemo(() => {
    return edges.filter((edge) => {
      const srcLabel = nodeLabel(edge.source, nodes);
      const tgtLabel = nodeLabel(edge.target, nodes);
      const matchesSearch =
        !searchTerm ||
        srcLabel.toLowerCase().includes(searchTerm.toLowerCase()) ||
        tgtLabel.toLowerCase().includes(searchTerm.toLowerCase()) ||
        edge.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (edge.details && edge.details.toLowerCase().includes(searchTerm.toLowerCase()));
      if (!matchesSearch) return false;
      if (!showInferred && edge.is_inferred) return false;
      const conf = edge.confidence ?? 0.7;
      if (conf < minConfidence) return false;
      if (effectiveSource !== "all" && edge.source_document !== effectiveSource) return false;
      return true;
    });
  }, [edges, nodes, searchTerm, showInferred, minConfidence, effectiveSource]);

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between gap-4 mb-6">
        <h3 className="text-lg font-semibold">
          Causal Relationships {sourceLabel ? `— ${sourceLabel}` : ""}
        </h3>
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search relationships..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <div className="flex items-center gap-6 mb-4 p-4 bg-muted/50 rounded-lg flex-wrap">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Filters:</span>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            id="show-inferred-causal"
            checked={showInferred}
            onCheckedChange={setShowInferred}
          />
          <Label htmlFor="show-inferred-causal" className="text-sm cursor-pointer">
            Show Inferred
          </Label>
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-sm whitespace-nowrap">Min confidence:</Label>
          <Input
            type="number"
            min={0}
            max={1}
            step={0.1}
            value={minConfidence}
            onChange={(e) => setMinConfidence(parseFloat(e.target.value) || 0)}
            className="w-20"
          />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-sm whitespace-nowrap">Source:</Label>
          <Select
        value={effectiveSource}
        onValueChange={setSelectedSource}
      >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="All sources" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sources</SelectItem>
              {availableSources.map((s) => (
                <SelectItem key={s} value={s}>
                  {s.length > 40 ? `${s.substring(0, 40)}...` : s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {filteredEdges.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <p className="text-muted-foreground mb-2">
            {searchTerm || effectiveSource !== "all" || !showInferred || minConfidence > 0
              ? "No relationships match your filters"
              : "No causal relationships in this source"}
          </p>
          <p className="text-sm text-muted-foreground">
            Use Causal Graph to view the graph; upload a CSV for data-driven relationships.
          </p>
        </div>
      ) : (
        <div className="border rounded-md overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[18%]">Source</TableHead>
                <TableHead className="w-[12%]">Relationship</TableHead>
                <TableHead className="w-[18%]">Target</TableHead>
                <TableHead className="w-[8%]">Confidence</TableHead>
                <TableHead className="w-[8%]">Type</TableHead>
                <TableHead className="w-[12%]">Document</TableHead>
                <TableHead className="w-[24%]">Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredEdges.map((edge) => (
                <TableRow key={edge.id}>
                  <TableCell className="font-medium">
                    {nodeLabel(edge.source, nodes)}
                  </TableCell>
                  <TableCell className="text-primary font-medium">{edge.label}</TableCell>
                  <TableCell className="font-medium">
                    {nodeLabel(edge.target, nodes)}
                  </TableCell>
                  <TableCell>{(edge.confidence ?? 0.7).toFixed(2)}</TableCell>
                  <TableCell>
                    {edge.is_inferred ? (
                      <Badge variant="secondary" className="text-xs">Inferred</Badge>
                    ) : (
                      <Badge variant="outline" className="text-xs">Direct</Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm max-w-[140px] truncate">
                    {edge.source_document ?? "—"}
                  </TableCell>
                  <TableCell>
                    {edge.details ? (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-6 w-6">
                            <Info className="h-4 w-4 text-muted-foreground" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-md">
                          <p className="text-sm whitespace-pre-wrap">{edge.details}</p>
                        </TooltipContent>
                      </Tooltip>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </Card>
  );
}
