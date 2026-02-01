import { useState, useEffect, useCallback } from "react";
import { useLocation } from "wouter";
import { KnowledgeGraphVisualization } from "@/components/KnowledgeGraphVisualization";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { hfApi } from "@/lib/api-client";
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
import type { GraphNode, GraphEdge } from "@shared/schema";

type CausalSource = { id: string; label: string; type: string };

export default function CausalGraphPage() {
  const { toast } = useToast();
  const [location] = useLocation();
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [includeInferred, setIncludeInferred] = useState(true);
  const [sources, setSources] = useState<CausalSource[]>([
    { id: "kb", label: "Causal graph from knowledge base", type: "kb" },
  ]);
  const [selectedSource, setSelectedSource] = useState<string>("kb");
  const [sourceLabel, setSourceLabel] = useState<string>("Causal graph from knowledge base");
  const [dowhyTreatment, setDowhyTreatment] = useState<string>("");
  const [dowhyOutcome, setDowhyOutcome] = useState<string>("");
  const [dowhyResult, setDowhyResult] = useState<{
    estimate_value: number | null;
    estimate: string;
    refutation: string;
  } | null>(null);
  const [dowhyLoading, setDowhyLoading] = useState(false);

  // Load causal graph data (KB or dataset)
  const loadCausalGraph = useCallback(async () => {
    setIsLoading(true);
    try {
      const isDataset = selectedSource !== "kb";
      const result = await hfApi.getCausalGraph(
        includeInferred,
        isDataset ? "dataset" : "kb",
        isDataset ? selectedSource : undefined
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

        toast({
          title: "Causal Graph Loaded",
          description: `Found ${graphNodes.length} nodes and ${graphEdges.length} relationships`,
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
  }, [selectedSource, includeInferred, toast]);

  // Fetch sources from backend (KB + data-driven datasets)
  const fetchSources = useCallback(async () => {
    const res = await hfApi.getCausalGraphSources();
    if (res.success && res.data?.sources?.length) {
      setSources(res.data.sources);
      const kb = res.data.sources.find((s: CausalSource) => s.type === "kb");
      if (kb) setSourceLabel(kb.label);
    }
  }, []);

  // Load sources on mount and when page is visited
  useEffect(() => {
    if (location === "/causal-graph") fetchSources();
  }, [location, fetchSources]);

  // Load graph when page is visited or source/inferred changes
  useEffect(() => {
    if (location === "/causal-graph") {
      const src = sources.find((s) => s.id === selectedSource);
      if (src) setSourceLabel(src.label);
      loadCausalGraph();
      if (selectedSource === "kb") {
        setDowhyResult(null);
        setDowhyTreatment("");
        setDowhyOutcome("");
      }
    }
  }, [location, selectedSource, includeInferred, loadCausalGraph, sources]);

  // Refetch when user returns to tab (e.g. after restarting server) so clean state is shown
  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible" && location === "/causal-graph") {
        fetchSources();
        loadCausalGraph();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [location, fetchSources, loadCausalGraph]);

  const handleExportCurrent = useCallback(async () => {
    try {
      toast({ title: "Exporting causal graph...", description: "Preparing export." });
      const isDataset = selectedSource !== "kb";
      const result = await hfApi.exportCausalGraph(
        isDataset ? "dataset" : "kb",
        isDataset ? selectedSource : undefined,
        includeInferred
      );
      if (result.success && result.data) {
        const blob = new Blob([JSON.stringify(result.data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `causal-graph-${isDataset ? selectedSource.replace(/\s+/g, "-") : "kb"}-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        const n = result.data.nodes?.length ?? 0;
        const e = result.data.edges?.length ?? 0;
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
  }, [selectedSource, includeInferred, toast]);

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
              Causal graph: from knowledge base or data-driven (CSV) via causal-learn.
              <span className="block text-xs mt-1">Knowledge base and uploaded sources are cleared on server restart. After restarting the server, refresh this page (or click Refresh sources) to see the clean graph. After uploading a CSV, click Refresh sources to see it in the list.</span>
            </p>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <Label htmlFor="source-select" className="text-sm text-muted-foreground">
                Source
              </Label>
              <Select
                value={selectedSource}
                onValueChange={(v) => setSelectedSource(v)}
              >
                <SelectTrigger id="source-select" className="w-[220px]">
                  <SelectValue placeholder="Select source" />
                </SelectTrigger>
                <SelectContent>
                  {sources.map((s) => (
                    <SelectItem key={s.id} value={s.id}>
                      {s.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {selectedSource === "kb" && (
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
              onClick={() => { fetchSources(); loadCausalGraph(); }}
              title="Update the list after uploading a CSV"
            >
              Refresh sources
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
            {selectedSource === "kb"
              ? "No causal relationships found in the knowledge base."
              : "No data-driven causal graph for this dataset."}
          </p>
          <p className="text-sm text-muted-foreground">
            {selectedSource === "kb"
              ? "Upload documents with causal language (e.g., \"X causes Y\", \"A leads to B\") or upload a CSV to see data-driven causal graphs."
              : "Upload a CSV file to run causal discovery (PC algorithm via causal-learn) and see the graph here."}
          </p>
        </Card>
      ) : (
        <>
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

          {selectedSource !== "kb" && nodes.length > 0 && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-2">Causal inference (DoWhy)</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Estimate the causal effect of a treatment variable on an outcome using{" "}
                <a href="https://github.com/py-why/dowhy" target="_blank" rel="noopener noreferrer" className="underline">DoWhy</a>{" "}
                on the discovered graph and data.
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
                      const res = await hfApi.runDoWhyEffect(selectedSource, dowhyTreatment, dowhyOutcome);
                      if (res.success && res.data) {
                        setDowhyResult({
                          estimate_value: res.data.estimate_value ?? null,
                          estimate: res.data.estimate ?? "",
                          refutation: res.data.refutation ?? "",
                        });
                        toast({ title: "Effect estimated", description: `Effect of ${dowhyTreatment} on ${dowhyOutcome}` });
                      } else {
                        toast({ title: "DoWhy failed", description: res.error, variant: "destructive" });
                      }
                    } catch (e) {
                      toast({ title: "DoWhy failed", description: String(e), variant: "destructive" });
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
                <div className="space-y-3 rounded-lg border p-4 bg-muted/30">
                  <div className="flex items-baseline gap-2">
                    <span className="text-muted-foreground">Estimated effect:</span>
                    <span className="font-mono font-semibold">
                      {dowhyResult.estimate_value != null ? dowhyResult.estimate_value.toFixed(4) : "—"}
                    </span>
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground">Details: </span>
                    <span className="font-mono text-xs break-all">{dowhyResult.estimate}</span>
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground">Refutation: </span>
                    <span className="font-mono text-xs">{dowhyResult.refutation}</span>
                  </div>
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
                <strong>Causal graph from dataset (CSV):</strong> When you upload a CSV, causal discovery is run using
                the <a href="https://github.com/py-why/causal-learn" target="_blank" rel="noopener noreferrer" className="underline">causal-learn</a> library
                (PC algorithm). Below the graph, <a href="https://github.com/py-why/dowhy" target="_blank" rel="noopener noreferrer" className="underline">DoWhy</a> is used to estimate causal effects (treatment → outcome) on the discovered graph and data.
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
