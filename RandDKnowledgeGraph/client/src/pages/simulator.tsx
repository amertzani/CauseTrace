import { useState, useEffect } from "react";
import { useKnowledgeStore } from "@/lib/knowledge-store";
import { hfApi } from "@/lib/api-client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Play, Trash2, CheckCircle2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface ExperimentConfig {
  name: string;
  description: string;
  inputData: string;
  scenarioType: "hypothesis" | "prediction" | "what_if" | "validation";
  targetNodes?: string[];
  parameters?: Record<string, any>;
}

interface ExperimentResult {
  id: string;
  config: ExperimentConfig;
  status: "running" | "completed" | "failed";
  results?: any;
  error?: string;
  timestamp: string;
}

export default function SimulatorPage() {
  const { toast } = useToast();
  const { facts, nodes, edges } = useKnowledgeStore();
  const [experiments, setExperiments] = useState<ExperimentResult[]>([]);
  const [currentExperiment, setCurrentExperiment] = useState<ExperimentConfig>({
    name: "",
    description: "",
    inputData: "",
    scenarioType: "hypothesis",
    targetNodes: [],
    parameters: {},
  });
  const [isRunning, setIsRunning] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);

  // Load saved experiments on mount
  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      const result = await hfApi.getExperiments();
      if (result.success && result.data?.experiments) {
        setExperiments(result.data.experiments);
      }
    } catch (error) {
      console.error("Failed to load experiments:", error);
    }
  };

  const handleRunExperiment = async () => {
    if (!currentExperiment.name || !currentExperiment.inputData) {
      toast({
        title: "Validation Error",
        description: "Please provide a name and input data for the experiment",
        variant: "destructive",
      });
      return;
    }

    setIsRunning(true);
    try {
      const result = await hfApi.runExperiment(currentExperiment);
      
      if (result.success) {
        const newExperiment: ExperimentResult = {
          id: result.data?.id || Date.now().toString(),
          config: currentExperiment,
          status: "completed",
          results: result.data?.results,
          timestamp: new Date().toISOString(),
        };
        
        setExperiments(prev => [newExperiment, ...prev]);
        setSelectedExperiment(newExperiment.id);
        
        toast({
          title: "Experiment Completed",
          description: `Experiment "${currentExperiment.name}" has been completed successfully`,
        });
        
        // Reset form
        setCurrentExperiment({
          name: "",
          description: "",
          inputData: "",
          scenarioType: "hypothesis",
          targetNodes: [],
          parameters: {},
        });
      } else {
        throw new Error(result.error || "Failed to run experiment");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      toast({
        title: "Experiment Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsRunning(false);
    }
  };

  const handleTestScenario = async (scenarioType: string, inputData: string) => {
    setIsRunning(true);
    try {
      const result = await hfApi.testScenario({
        scenarioType,
        inputData,
        knowledgeGraph: { facts, nodes, edges },
      });
      
      if (result.success) {
        toast({
          title: "Scenario Tested",
          description: "Scenario has been tested against the knowledge graph",
        });
        return result.data;
      } else {
        throw new Error(result.error || "Failed to test scenario");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      toast({
        title: "Scenario Test Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsRunning(false);
    }
  };

  const handleDeleteExperiment = async (experimentId: string) => {
    try {
      const result = await hfApi.deleteExperiment(experimentId);
      if (result.success) {
        setExperiments(prev => prev.filter(exp => exp.id !== experimentId));
        if (selectedExperiment === experimentId) {
          setSelectedExperiment(null);
        }
        toast({
          title: "Experiment Deleted",
          description: "Experiment has been deleted successfully",
        });
      }
    } catch (error) {
      toast({
        title: "Delete Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    }
  };

  const selectedExperimentData = experiments.find(exp => exp.id === selectedExperiment);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold mb-2">Simulator</h1>
        <p className="text-muted-foreground">
          Run experiments and test scenarios based on your knowledge graph
        </p>
      </div>

      <Tabs defaultValue="create" className="space-y-4">
        <TabsList>
          <TabsTrigger value="create">Create Experiment</TabsTrigger>
          <TabsTrigger value="scenarios">Test Scenarios</TabsTrigger>
          <TabsTrigger value="history">Experiment History</TabsTrigger>
        </TabsList>

        <TabsContent value="create" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>New Experiment</CardTitle>
              <CardDescription>
                Configure and run an experiment using data from your knowledge graph
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="experiment-name">Experiment Name</Label>
                <Input
                  id="experiment-name"
                  placeholder="e.g., Drug Interaction Analysis"
                  value={currentExperiment.name}
                  onChange={(e) =>
                    setCurrentExperiment({ ...currentExperiment, name: e.target.value })
                  }
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="experiment-description">Description</Label>
                <Textarea
                  id="experiment-description"
                  placeholder="Describe what this experiment aims to test or discover..."
                  value={currentExperiment.description}
                  onChange={(e) =>
                    setCurrentExperiment({ ...currentExperiment, description: e.target.value })
                  }
                  rows={3}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="scenario-type">Scenario Type</Label>
                <Select
                  value={currentExperiment.scenarioType}
                  onValueChange={(value: any) =>
                    setCurrentExperiment({ ...currentExperiment, scenarioType: value })
                  }
                >
                  <SelectTrigger id="scenario-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="hypothesis">Hypothesis Testing</SelectItem>
                    <SelectItem value="prediction">Prediction</SelectItem>
                    <SelectItem value="what_if">What-If Analysis</SelectItem>
                    <SelectItem value="validation">Validation</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="input-data">Input Data (JSON or Text)</Label>
                <Textarea
                  id="input-data"
                  placeholder='{"entity": "Drug A", "property": "interaction", "value": "inhibits"}'
                  value={currentExperiment.inputData}
                  onChange={(e) =>
                    setCurrentExperiment({ ...currentExperiment, inputData: e.target.value })
                  }
                  rows={6}
                  className="font-mono text-sm"
                />
                <p className="text-xs text-muted-foreground">
                  Enter data in JSON format or plain text describing the experiment input
                </p>
              </div>

              <div className="space-y-2">
                <Label>Target Nodes (Optional)</Label>
                <Select
                  value=""
                  onValueChange={(value) => {
                    if (value && !currentExperiment.targetNodes?.includes(value)) {
                      setCurrentExperiment({
                        ...currentExperiment,
                        targetNodes: [...(currentExperiment.targetNodes || []), value],
                      });
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select nodes from knowledge graph" />
                  </SelectTrigger>
                  <SelectContent>
                    {nodes.slice(0, 50).map((node) => (
                      <SelectItem key={node.id} value={node.label}>
                        {node.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {currentExperiment.targetNodes && currentExperiment.targetNodes.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {currentExperiment.targetNodes.map((node) => (
                      <Badge
                        key={node}
                        variant="secondary"
                        className="cursor-pointer"
                        onClick={() => {
                          setCurrentExperiment({
                            ...currentExperiment,
                            targetNodes: currentExperiment.targetNodes?.filter((n) => n !== node),
                          });
                        }}
                      >
                        {node} ×
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              <Button
                onClick={handleRunExperiment}
                disabled={isRunning || !currentExperiment.name || !currentExperiment.inputData}
                className="w-full"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running Experiment...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Run Experiment
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scenarios" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Hypothesis Testing</CardTitle>
                <CardDescription>
                  Test a hypothesis against your knowledge graph
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Enter your hypothesis..."
                  rows={4}
                  id="hypothesis-input"
                />
                <Button
                  onClick={async () => {
                    const input = (document.getElementById("hypothesis-input") as HTMLTextAreaElement)?.value;
                    if (input) {
                      await handleTestScenario("hypothesis", input);
                    }
                  }}
                  disabled={isRunning}
                  className="w-full"
                >
                  Test Hypothesis
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>What-If Analysis</CardTitle>
                <CardDescription>
                  Explore what would happen if certain conditions changed
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Enter your what-if scenario..."
                  rows={4}
                  id="whatif-input"
                />
                <Button
                  onClick={async () => {
                    const input = (document.getElementById("whatif-input") as HTMLTextAreaElement)?.value;
                    if (input) {
                      await handleTestScenario("what_if", input);
                    }
                  }}
                  disabled={isRunning}
                  className="w-full"
                >
                  Run What-If
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Prediction</CardTitle>
                <CardDescription>
                  Make predictions based on existing knowledge
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Enter data for prediction..."
                  rows={4}
                  id="prediction-input"
                />
                <Button
                  onClick={async () => {
                    const input = (document.getElementById("prediction-input") as HTMLTextAreaElement)?.value;
                    if (input) {
                      await handleTestScenario("prediction", input);
                    }
                  }}
                  disabled={isRunning}
                  className="w-full"
                >
                  Make Prediction
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Validation</CardTitle>
                <CardDescription>
                  Validate data or relationships against the knowledge graph
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Enter data to validate..."
                  rows={4}
                  id="validation-input"
                />
                <Button
                  onClick={async () => {
                    const input = (document.getElementById("validation-input") as HTMLTextAreaElement)?.value;
                    if (input) {
                      await handleTestScenario("validation", input);
                    }
                  }}
                  disabled={isRunning}
                  className="w-full"
                >
                  Validate
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="history" className="space-y-4">
          {experiments.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground">
                No experiments have been run yet. Create your first experiment to get started.
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">Experiment List</h3>
                <div className="space-y-2">
                  {experiments.map((experiment) => (
                    <Card
                      key={experiment.id}
                      className={`cursor-pointer transition-colors ${
                        selectedExperiment === experiment.id
                          ? "border-primary"
                          : "hover:border-primary/50"
                      }`}
                      onClick={() => setSelectedExperiment(experiment.id)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-semibold">{experiment.config.name}</h4>
                            <p className="text-sm text-muted-foreground line-clamp-1">
                              {experiment.config.description}
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge variant="outline">
                                {experiment.config.scenarioType}
                              </Badge>
                              {experiment.status === "completed" ? (
                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                              ) : experiment.status === "failed" ? (
                                <XCircle className="h-4 w-4 text-red-500" />
                              ) : (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              )}
                            </div>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteExperiment(experiment.id);
                            }}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>

              {selectedExperimentData && (
                <Card>
                  <CardHeader>
                    <CardTitle>Experiment Results</CardTitle>
                    <CardDescription>
                      {selectedExperimentData.config.name}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label>Status</Label>
                      <Badge
                        variant={
                          selectedExperimentData.status === "completed"
                            ? "default"
                            : selectedExperimentData.status === "failed"
                            ? "destructive"
                            : "secondary"
                        }
                      >
                        {selectedExperimentData.status}
                      </Badge>
                    </div>

                    {selectedExperimentData.results && (
                      <div>
                        <Label>Results</Label>
                        <pre className="mt-2 p-4 bg-muted rounded-md text-sm overflow-auto max-h-96">
                          {JSON.stringify(selectedExperimentData.results, null, 2)}
                        </pre>
                      </div>
                    )}

                    {selectedExperimentData.error && (
                      <Alert variant="destructive">
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{selectedExperimentData.error}</AlertDescription>
                      </Alert>
                    )}

                    <div className="text-sm text-muted-foreground">
                      <p>Type: {selectedExperimentData.config.scenarioType}</p>
                      <p>Run at: {new Date(selectedExperimentData.timestamp).toLocaleString()}</p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
