import { useState, useEffect, useCallback } from "react";
import { useLocation } from "wouter";
import { KnowledgeGraphVisualization } from "@/components/KnowledgeGraphVisualization";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { hfApi, getUploadedDocNamesSession } from "@/lib/api-client";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Loader2, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { GraphNode, GraphEdge } from "@shared/schema";

type CausalSource = { id: string; label: string; type: string };
type SourceMode = "kb" | "data_only" | "both";

/** Build a plain-language interpretation when the backend doesn't send one. */
function formatEffectExplanation(
  treatment: string,
  outcome: string,
  value: number | null
): string {
  if (value == null) return "No numeric effect was estimated.";
  const t = treatment.replace(/_/g, " ");
  const o = outcome.replace(/_/g, " ");
  const absVal = Math.abs(value);
  if (absVal < 1e-6) {
    return `An effect near zero suggests little or no linear causal effect of "${t}" on "${o}" in this model. Changes in the treatment are not associated with systematic changes in the outcome.`;
  }
  if (value > 0) {
    return `A positive effect (${value.toFixed(4)}) means that an increase in "${t}" tends to increase "${o}". On average, a one-unit increase in the treatment is associated with an increase of about ${absVal.toFixed(4)} units in the outcome.`;
  }
  return `A negative effect (${value.toFixed(4)}) means that an increase in "${t}" tends to decrease "${o}". On average, a one-unit increase in the treatment is associated with a decrease of about ${absVal.toFixed(4)} units in the outcome.`;
}

export default function CausalGraphPage() {
  const { toast } = useToast();
  const [location] = useLocation();
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [graphWarning, setGraphWarning] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [includeInferred, setIncludeInferred] = useState(true);
  const [sourceMode, setSourceMode] = useState<SourceMode>("kb");
  const [sources, setSources] = useState<CausalSource[]>([
    { id: "kb", label: "Causal graph from knowledge base", type: "kb" },
  ]);
  const [selectedDataset, setSelectedDataset] = useState<string>("all");
  const [sourceLabel, setSourceLabel] = useState<string>("Causal graph from knowledge base");
  const [dowhyTreatment, setDowhyTreatment] = useState<string>("");
  const [dowhyOutcome, setDowhyOutcome] = useState<string>("");
  const [dowhyResult, setDowhyResult] = useState<{
    estimate_value: number | null;
    interpretation: string | null;
    estimate: string;
    refutation: string;
  } | null>(null);
  const [dowhyLoading, setDowhyLoading] = useState(false);

  // Load causal graph data (kb only, data only, both, or one dataset)
  const loadCausalGraph = useCallback(async () => {
    setIsLoading(true);
    setGraphWarning(null);
    try {
      let apiSource: "kb" | "dataset" | "data_only" | "both" = "kb";
      const documentName = selectedDataset === "all" ? undefined : selectedDataset;
      if (sourceMode === "kb") {
        apiSource = "kb";
      } else if (sourceMode === "both") {
        apiSource = "both";
      } else {
        if (selectedDataset === "all") {
          apiSource = "data_only";
        } else {
          apiSource = "dataset";
        }
      }
      const result = await hfApi.getCausalGraph(
        includeInferred,
        apiSource,
        documentName
      );

      if (result.success && result.data) {
        const apiNodes = result.data.nodes || [];
        const apiEdges = result.data.edges || [];

        const graphNodes: GraphNode[] = apiNodes.map((node: any) => ({
          id: node.id,
          label: node.label,
          type: node.type || "entity",
          connections: node.connections || 0,
        }));

        const graphEdges: GraphEdge[] = apiEdges.map((edge: any) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.label,
          details: edge.details,
          isInferred: edge.is_inferred || false,
        }));

        setNodes(graphNodes);
        setEdges(graphEdges);
        setStats(result.data.stats || {});
        setGraphWarning(result.data.empty_reason || result.data.warning || null);

        toast({
          title: "Causal Graph Loaded",
          description: result.data.warning ? result.data.warning : `Found ${graphNodes.length} nodes and ${graphEdges.length} relationships`,
        });
      } else {
        throw new Error(result.error || "Failed to load causal graph");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      toast({
        title: "Failed to Load Causal Graph",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [sourceMode, selectedDataset, includeInferred, toast]);

  // Fetch sources from backend (KB + uploaded documents only; same as Documents page)
  const fetchSources = useCallback(async () => {
    const res = await hfApi.getCausalGraphSources();
    if (res.success && Array.isArray(res.data?.sources)) {
      setSources(res.data.sources);
      const kb = res.data.sources.find((s: CausalSource) => s.type === "kb");
      if (kb) setSourceLabel(kb.label);
    }
  }, []);

  // Load sources on mount and when page is visited (do not auto-load graph)
  useEffect(() => {
    if (location === "/causal-graph") fetchSources();
  }, [location, fetchSources]);

  // When upload completes elsewhere, refresh source list so new doc can be selected
  useEffect(() => {
    const onSourcesRefresh = () => {
      if (location === "/causal-graph") fetchSources();
    };
    window.addEventListener("sources-refresh", onSourcesRefresh);
    return () => window.removeEventListener("sources-refresh", onSourcesRefresh);
  }, [location, fetchSources]);

  // Refetch sources when user returns to tab (do not auto-load graph)
  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible" && location === "/causal-graph") fetchSources();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [location, fetchSources]);

  // Reset selected dataset if it's no longer in the session-uploaded list
  const sessionUploadedIds = getUploadedDocNamesSession();
  const availableDatasetIds = sources.filter((s) => s.type === "dataset" && sessionUploadedIds.includes(s.id)).map((s) => s.id);
  useEffect(() => {
    if (selectedDataset !== "all" && availableDatasetIds.length > 0 && !availableDatasetIds.includes(selectedDataset)) {
      setSelectedDataset("all");
    }
  }, [sources, selectedDataset, sessionUploadedIds.join(",")]);

  const handleExportCurrent = useCallback(async () => {
    try {
      toast({ title: "Exporting causal graph...", description: "Preparing export." });
      let exportSource: "kb" | "dataset" | "all" = "kb";
      let documentName: string | undefined;
      if (sourceMode === "both") {
        exportSource = "all";
      } else if (sourceMode === "data_only" && selectedDataset !== "all") {
        exportSource = "dataset";
        documentName = selectedDataset;
      } else if (sourceMode === "data_only") {
        exportSource = "all";
      }
      const result = await hfApi.exportCausalGraph(
        exportSource === "dataset" ? "dataset" : exportSource === "all" ? "all" : "kb",
        documentName,
        includeInferred
      );
      if (result.success && result.data) {
        const data = result.data as any;
        let n = data.nodes?.length ?? 0;
        let e = data.edges?.length ?? 0;
        if (n === 0 && data.kb) {
          n = (data.kb.nodes?.length ?? 0) + Object.values(data.datasets || {}).reduce((s: number, v: any) => s + (v.nodes?.length ?? 0), 0);
          e = (data.kb.edges?.length ?? 0) + Object.values(data.datasets || {}).reduce((s: number, v: any) => s + (v.edges?.length ?? 0), 0);
        }
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `causal-graph-${sourceMode}-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast({ title: "Export successful", description: `Exported ${n} nodes, ${e} edges.` });
      } else {
        throw new Error(result.error || "Export failed");
      }
    } catch (error) {
      toast({
        title: "Export failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    }
  }, [sourceMode, selectedDataset, includeInferred, toast]);

  const handleNodeClick = (node: GraphNode) => {
    toast({
      title: "Entity selected",
      description: `${node.label} (${node.connections} causal connections)`,
    });
  };

  const handleNodeEdit = (node: GraphNode) => {
    toast({
      title: "Read-only",
      description: "Causal graph is generated from knowledge base or CSV data",
    });
  };

  const handleNodeMove = (nodeId: string, position: { x: number; y: number; z: number }) => {
    // Allow node positioning for better visualization
    setNodes(prev => prev.map(n => 
      n.id === nodeId ? { ...n } : n
    ));
  };

  const handleEdgeEdit = (edge: GraphEdge) => {
    toast({
      title: "Read-only",
      description: "Causal relationships are extracted from knowledge base or CSV",
    });
  };

  const handleEdgeDelete = async (edgeId: string) => {
    toast({
      title: "Read-only",
      description: "Causal relationships are automatically extracted",
    });
  };

  const handleConnectionCreate = () => {
    toast({
      title: "Read-only",
      description: "Add causal facts to the knowledge base or upload a CSV to see causal relationships",
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold mb-2">Causal Graph</h1>
            <p className="text-muted-foreground">
              Causal graph from your knowledge base or from uploaded data. Select a source and click Load graph.
            </p>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex flex-col gap-2">
              <span className="text-sm font-medium text-muted-foreground">Show relationships from</span>
              <RadioGroup
                value={sourceMode}
                onValueChange={(v) => setSourceMode(v as SourceMode)}
                className="flex flex-wrap gap-4"
              >
                <label
                  htmlFor="mode-kb"
                  className="flex items-center space-x-2 cursor-pointer select-none rounded p-1 hover:bg-muted/50"
                >
                  <RadioGroupItem value="kb" id="mode-kb" />
                  <span>Knowledge graph only</span>
                </label>
                <label
                  htmlFor="mode-data"
                  className="flex items-center space-x-2 cursor-pointer select-none rounded p-1 hover:bg-muted/50"
                >
                  <RadioGroupItem value="data_only" id="mode-data" />
                  <span>Data only (CSV)</span>
                </label>
                <label
                  htmlFor="mode-both"
                  className="flex items-center space-x-2 cursor-pointer select-none rounded p-1 hover:bg-muted/50"
                >
                  <RadioGroupItem value="both" id="mode-both" />
                  <span>Both</span>
                </label>
              </RadioGroup>
            </div>
            <div className="flex items-center gap-2">
              <Label htmlFor="dataset-source" className="text-sm text-muted-foreground">Dataset</Label>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger id="dataset-source" className="w-[200px]">
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
            {sourceMode === "kb" && (
              <div className="flex items-center gap-2">
                <Switch
                  id="include-inferred"
                  checked={includeInferred}
                  onCheckedChange={setIncludeInferred}
                />
                <Label htmlFor="include-inferred" className="cursor-pointer">
                  Include Inferred
                </Label>
              </div>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={fetchSources}
              title="Update the list after uploading a document"
            >
              Refresh sources
            </Button>
            <Button
              variant="default"
              size="sm"
              onClick={() => {
                if (sourceMode !== "kb") {
                  setDowhyResult(null);
                  setDowhyTreatment("");
                  setDowhyOutcome("");
                }
                loadCausalGraph();
              }}
              disabled={isLoading}
              title="Load causal graph for the selected source"
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
              Load graph
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportCurrent}
              title="Download current causal graph as JSON"
            >
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
            <Badge variant="outline">
              {nodes.length} nodes
            </Badge>
            <Badge variant="outline">
              {edges.length} edges
            </Badge>
          </div>
        </div>
      </div>

      {isLoading ? (
        <Card className="p-12 text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading causal graph...</p>
        </Card>
      ) : nodes.length === 0 ? (
        <Card className="p-12 text-center">
          <p className="text-muted-foreground mb-4">
            No graph loaded yet. Select a source above and click &quot;Load graph&quot; to display it.
          </p>
          <p className="text-sm text-muted-foreground max-w-lg mx-auto">
            {sourceMode === "kb"
              ? "Upload documents (Upload) to add causal facts to the knowledge base, or add facts in the Knowledge Base. Then click \"Load graph\" above."
              : sourceMode === "data_only"
                ? "Upload a CSV (Upload) with numeric columns. Causal discovery runs automatically. After upload, click \"Refresh sources\" so the dataset appears in the Dataset dropdown, then click \"Load graph\"."
                : "Upload documents or CSV datasets, then Refresh sources and Load graph."}
          </p>
          <Button
            variant="default"
            size="sm"
            className="mt-3"
            onClick={() => loadCausalGraph()}
          >
            Load graph
          </Button>
          {sourceMode === "data_only" && availableDatasetIds.length === 0 && (
            <Button
              variant="outline"
              size="sm"
              className="mt-3 ml-2"
              onClick={() => fetchSources()}
            >
              Refresh sources
            </Button>
          )}
        </Card>
      ) : (
        <>
          {graphWarning && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/50 px-4 py-2 text-sm text-amber-800 dark:text-amber-200">
              {graphWarning}
            </div>
          )}
          <KnowledgeGraphVisualization
            nodes={nodes}
            edges={edges}
            graphTitle="Causal Graph"
            onNodeClick={handleNodeClick}
            onNodeEdit={handleNodeEdit}
            onNodeMove={handleNodeMove}
            onEdgeEdit={handleEdgeEdit}
            onEdgeDelete={handleEdgeDelete}
            onConnectionCreate={handleConnectionCreate}
          />

          {sourceMode !== "kb" && nodes.length > 0 && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-2">Causal inference</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Estimate the causal effect of a treatment variable on an outcome using the discovered graph and data.
              </p>
              <div className="flex flex-wrap items-end gap-4 mb-4">
                <div className="flex flex-col gap-2">
                  <Label>Treatment (cause)</Label>
                  <Select value={dowhyTreatment} onValueChange={(v) => { setDowhyTreatment(v); setDowhyResult(null); }}>
                    <SelectTrigger className="w-[200px]">
                      <SelectValue placeholder="Select variable" />
                    </SelectTrigger>
                    <SelectContent>
                      {nodes.map((n) => (
                        <SelectItem key={n.id} value={n.label}>{n.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex flex-col gap-2">
                  <Label>Outcome (effect)</Label>
                  <Select value={dowhyOutcome} onValueChange={(v) => { setDowhyOutcome(v); setDowhyResult(null); }}>
                    <SelectTrigger className="w-[200px]">
                      <SelectValue placeholder="Select variable" />
                    </SelectTrigger>
                    <SelectContent>
                      {nodes.map((n) => (
                        <SelectItem key={n.id} value={n.label}>{n.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  disabled={!dowhyTreatment || !dowhyOutcome || dowhyLoading}
                  onClick={async () => {
                    setDowhyLoading(true);
                    setDowhyResult(null);
                    try {
                      const dowhyDoc = sourceMode === "data_only" && selectedDataset !== "all" ? selectedDataset : (sources.find((s) => s.type === "dataset")?.id ?? "");
                      const res = dowhyDoc ? await hfApi.runDoWhyEffect(dowhyDoc, dowhyTreatment, dowhyOutcome) : { success: false, error: "Select a single dataset" };
                      if (res.success && res.data) {
                        setDowhyResult({
                          estimate_value: res.data.estimate_value ?? null,
                          interpretation: res.data.interpretation ?? null,
                          estimate: res.data.estimate ?? "",
                          refutation: res.data.refutation ?? "",
                        });
                        toast({ title: "Effect estimated", description: `Effect of ${dowhyTreatment} on ${dowhyOutcome}` });
                      } else {
                        toast({ title: "Effect estimation failed", description: res.error, variant: "destructive" });
                      }
                    } catch (e) {
                      toast({ title: "Effect estimation failed", description: String(e), variant: "destructive" });
                    } finally {
                      setDowhyLoading(false);
                    }
                  }}
                >
                  {dowhyLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                  Estimate causal effect
                </Button>
              </div>
              {dowhyResult && (
                <div className="space-y-4 rounded-lg border p-4 bg-muted/30">
                  <div className="flex items-baseline gap-2">
                    <span className="text-muted-foreground">Estimated effect:</span>
                    <span className="font-mono font-semibold text-lg">
                      {dowhyResult.estimate_value != null ? dowhyResult.estimate_value.toFixed(4) : "—"}
                    </span>
                  </div>
                  <div className="rounded-md border-l-4 border-primary bg-primary/5 p-3">
                    <p className="text-sm font-medium text-foreground mb-1">What this means</p>
                    <p className="text-sm text-foreground leading-relaxed">
                      {dowhyResult.interpretation ||
                        formatEffectExplanation(
                          dowhyTreatment,
                          dowhyOutcome,
                          dowhyResult.estimate_value
                        )}
                    </p>
                  </div>
                  <Collapsible>
                    <CollapsibleTrigger asChild>
                      <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
                        Technical details (estimand & refutation)
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="space-y-2 pt-2">
                      <div className="text-sm">
                        <span className="text-muted-foreground">Details: </span>
                        <span className="font-mono text-xs break-all block mt-1">{dowhyResult.estimate}</span>
                      </div>
                      <div className="text-sm">
                        <span className="text-muted-foreground">Refutation: </span>
                        <span className="font-mono text-xs block mt-1">{dowhyResult.refutation}</span>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                </div>
              )}
            </Card>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Causal Graph Statistics</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Entities</span>
                  <span className="font-semibold">{stats?.total_nodes || nodes.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Direct Relationships</span>
                  <span className="font-semibold">{stats?.direct_relationships || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Inferred Relationships</span>
                  <span className="font-semibold text-blue-600">
                    {stats?.inferred_relationships || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Relationships</span>
                  <span className="font-semibold">{stats?.total_edges || edges.length}</span>
                </div>
              </div>
            </Card>

            {stats?.most_causal_entities && stats.most_causal_entities.length > 0 && (
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4">Most Causal Entities</h3>
                <div className="space-y-2">
                  {stats.most_causal_entities.slice(0, 5).map((item: any, idx: number) => (
                    <div key={idx} className="flex justify-between items-center">
                      <span className="text-sm truncate flex-1">{item.entity}</span>
                      <Badge variant="secondary" className="ml-2">
                        {item.connections}
                      </Badge>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>

          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-2">About Causal Graphs</h3>
            <div className="text-sm text-muted-foreground space-y-2">
              <p>
                <strong>Causal graph from knowledge base:</strong> Cause-and-effect relationships are extracted from
                the knowledge base (causes, leads to, results in, triggers, affects, influences, etc.).
                Inferred relationships (A→B→C ⇒ A→C) can be included.
              </p>
              <p>
                <strong>Causal graph from dataset (CSV):</strong> When you upload a CSV, causal discovery is run on the data.
                You can then estimate causal effects (treatment → outcome) on the discovered graph.
              </p>
              <p className="text-xs mt-4">
                <strong>Note:</strong> Graphs are read-only. Upload documents or CSVs to update them.
              </p>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
