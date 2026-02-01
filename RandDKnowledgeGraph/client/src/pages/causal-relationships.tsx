import { useState, useEffect, useCallback } from "react";
import { useLocation } from "wouter";
import { CausalRelationshipsTable, type CausalEdge } from "@/components/CausalRelationshipsTable";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { hfApi, getUploadedDocNamesSession } from "@/lib/api-client";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";

type CausalSource = { id: string; label: string; type: string };
type SourceMode = "kb" | "data_only" | "both";

export default function CausalRelationshipsPage() {
  const { toast } = useToast();
  const [location] = useLocation();
  const [sources, setSources] = useState<CausalSource[]>([
    { id: "kb", label: "Causal graph from knowledge base", type: "kb" },
  ]);
  const [sourceMode, setSourceMode] = useState<SourceMode>("kb");
  const [selectedDataset, setSelectedDataset] = useState<string>("all");
  const [sourceLabel, setSourceLabel] = useState<string>("Causal graph from knowledge base");
  const [nodes, setNodes] = useState<{ id: string; label: string }[]>([]);
  const [edges, setEdges] = useState<CausalEdge[]>([]);
  const [includeInferred, setIncludeInferred] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedDocNames, setUploadedDocNames] = useState<string[]>([]);

  const fetchUploadedDocs = useCallback(async () => {
    const sessionNames = getUploadedDocNamesSession();
    const res = await hfApi.getDocuments();
    if (res.success && Array.isArray(res.data?.documents)) {
      const apiNames = (res.data.documents as { name?: string }[])
        .map((d) => d.name)
        .filter((n): n is string => Boolean(n?.trim()));
      const names = apiNames.filter((n) => sessionNames.includes(n));
      setUploadedDocNames(names);
    } else {
      setUploadedDocNames([]);
    }
  }, []);

  const loadCausalData = useCallback(async () => {
    setIsLoading(true);
    try {
      let apiSource: "kb" | "dataset" | "data_only" | "both" = "kb";
      const documentName = selectedDataset === "all" ? undefined : selectedDataset;
      if (sourceMode === "kb") apiSource = "kb";
      else if (sourceMode === "both") apiSource = "both";
      else {
        if (selectedDataset === "all") apiSource = "data_only";
        else apiSource = "dataset";
      }
      const result = await hfApi.getCausalGraph(
        includeInferred,
        apiSource,
        documentName
      );
      if (result.success && result.data) {
        const apiNodes = result.data.nodes || [];
        const apiEdges = result.data.edges || [];
        setNodes(
          apiNodes.map((n: any) => ({
            id: n.id,
            label: n.label ?? n.id,
          }))
        );
        setEdges(
          apiEdges.map((e: any) => ({
            id: e.id,
            source: e.source,
            target: e.target,
            label: e.label ?? "",
            details: e.details,
            confidence: e.confidence,
            is_inferred: e.is_inferred,
            source_document: e.source_document,
          }))
        );
      } else {
        setNodes([]);
        setEdges([]);
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Failed to load causal data";
      toast({ title: "Error", description: msg, variant: "destructive" });
      setNodes([]);
      setEdges([]);
    } finally {
      setIsLoading(false);
    }
  }, [sourceMode, selectedDataset, includeInferred, toast]);

  const fetchSources = useCallback(async () => {
    const res = await hfApi.getCausalGraphSources();
    if (res.success && Array.isArray(res.data?.sources)) {
      setSources(res.data.sources);
      const kb = res.data.sources.find((s: CausalSource) => s.type === "kb");
      if (kb) setSourceLabel(kb.label);
    }
  }, []);

  useEffect(() => {
    if (location === "/causal-relationships") {
      fetchSources();
      fetchUploadedDocs();
    }
  }, [location, fetchSources, fetchUploadedDocs]);

  // When upload completes, refresh sources and document list so new doc can be selected
  useEffect(() => {
    const onSourcesRefresh = () => {
      if (location === "/causal-relationships") {
        fetchSources();
        fetchUploadedDocs();
      }
    };
    window.addEventListener("sources-refresh", onSourcesRefresh);
    return () => window.removeEventListener("sources-refresh", onSourcesRefresh);
  }, [location, fetchSources, fetchUploadedDocs]);

  // Update source label when mode/source changes (do not auto-load data)
  useEffect(() => {
    if (location === "/causal-relationships") {
      if (sourceMode === "kb") setSourceLabel(selectedDataset === "all" ? "Knowledge graph only" : `Knowledge graph: ${selectedDataset}`);
      else if (sourceMode === "both") setSourceLabel(selectedDataset === "all" ? "Knowledge graph + Data" : `Both: ${selectedDataset}`);
      else if (selectedDataset === "all") setSourceLabel("Data only (all datasets)");
      else {
        const src = sources.find((s) => s.id === selectedDataset);
        setSourceLabel(src ? src.label : "Data only");
      }
    }
  }, [location, sourceMode, selectedDataset, sources]);

  // Reset selected dataset if it's no longer in the session-uploaded list
  const sessionUploadedIds = getUploadedDocNamesSession();
  useEffect(() => {
    const ids = sources.filter((s) => s.type === "dataset" && sessionUploadedIds.includes(s.id)).map((s) => s.id);
    if (selectedDataset !== "all" && ids.length > 0 && !ids.includes(selectedDataset)) {
      setSelectedDataset("all");
    }
  }, [sources, selectedDataset, sessionUploadedIds.join(",")]);

  const directCount = edges.filter((e) => !e.is_inferred).length;
  const inferredCount = edges.filter((e) => e.is_inferred).length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-semibold mb-2">Causal Relationships</h1>
          <p className="text-muted-foreground">
            View extracted correlations and causes from the knowledge base or from uploaded data.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium">Show relationships from</span>
          <RadioGroup
            value={sourceMode}
            onValueChange={(v) => setSourceMode(v as SourceMode)}
            className="flex flex-wrap gap-4"
          >
            <label
              htmlFor="cr-mode-kb"
              className="flex items-center space-x-2 cursor-pointer select-none rounded p-1 hover:bg-muted/50"
            >
              <RadioGroupItem value="kb" id="cr-mode-kb" />
              <span>Knowledge graph only</span>
            </label>
            <label
              htmlFor="cr-mode-data"
              className="flex items-center space-x-2 cursor-pointer select-none rounded p-1 hover:bg-muted/50"
            >
              <RadioGroupItem value="data_only" id="cr-mode-data" />
              <span>Data only (CSV)</span>
            </label>
            <label
              htmlFor="cr-mode-both"
              className="flex items-center space-x-2 cursor-pointer select-none rounded p-1 hover:bg-muted/50"
            >
              <RadioGroupItem value="both" id="cr-mode-both" />
              <span>Both</span>
            </label>
          </RadioGroup>
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="cr-dataset-source" className="text-sm">Dataset</Label>
          <Select value={selectedDataset} onValueChange={setSelectedDataset}>
            <SelectTrigger id="cr-dataset-source" className="w-[200px]">
              <SelectValue placeholder="Dataset" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All datasets</SelectItem>
              {sources
                .filter((s) => s.type === "dataset" && getUploadedDocNamesSession().includes(s.id))
                .map((s) => (
                  <SelectItem key={s.id} value={s.id}>{s.label}</SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Include inferred:</span>
          <Button
            variant={includeInferred ? "default" : "outline"}
            size="sm"
            onClick={() => setIncludeInferred(true)}
          >
            Yes
          </Button>
          <Button
            variant={!includeInferred ? "default" : "outline"}
            size="sm"
            onClick={() => setIncludeInferred(false)}
          >
            No
          </Button>
        </div>
        <Button variant="outline" size="sm" onClick={() => { fetchSources(); fetchUploadedDocs(); }} title="Update the list after uploading a document">
          Refresh sources
        </Button>
        <Button variant="default" size="sm" onClick={() => loadCausalData()} disabled={isLoading} title="Load relationships for the selected source">
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
          Load relationships
        </Button>
      </div>

      {isLoading ? (
        <Card className="p-12 text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Loading relationships...</p>
        </Card>
      ) : edges.length === 0 ? (
        <Card className="p-12 text-center">
          <p className="text-muted-foreground mb-4">No relationships loaded yet. Select a source above and click &quot;Load relationships&quot; to display them.</p>
          <Button variant="default" size="sm" onClick={() => loadCausalData()}>
            Load relationships
          </Button>
        </Card>
      ) : (
        <>
          <div className="flex flex-wrap gap-4">
            <Card className="p-4">
              <div className="text-2xl font-semibold">{edges.length}</div>
              <div className="text-sm text-muted-foreground">Total relationships</div>
            </Card>
            <Card className="p-4">
              <div className="text-2xl font-semibold">{directCount}</div>
              <div className="text-sm text-muted-foreground">Direct</div>
            </Card>
            <Card className="p-4">
              <div className="text-2xl font-semibold">{inferredCount}</div>
              <div className="text-sm text-muted-foreground">
                Inferred <Badge variant="secondary" className="text-xs ml-1">from paths</Badge>
              </div>
            </Card>
          </div>
          <CausalRelationshipsTable
            edges={edges}
            nodes={nodes}
            sourceLabel={sourceLabel}
            allowedSourceDocuments={uploadedDocNames}
          />
        </>
      )}
    </div>
  );
}
