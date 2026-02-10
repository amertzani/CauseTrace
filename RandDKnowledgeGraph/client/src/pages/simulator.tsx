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
  scenarioDescription: string; // Natural language scenario description
  inputData?: string; // Optional extra data not in knowledge base
  scenarioType: "hypothesis" | "prediction" | "what_if" | "validation";
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
    scenarioDescription: "",
    inputData: "",
    scenarioType: "hypothesis",
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
    if (!currentExperiment.name || !currentExperiment.scenarioDescription) {
      toast({
        title: "Validation Error",
        description: "Please provide a name and scenario description for the experiment",
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
          scenarioDescription: "",
          inputData: "",
          scenarioType: "hypothesis",
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
                <Label htmlFor="scenario-description">
                  Scenario Description <span className="text-red-500">*</span>
                </Label>
                <Textarea
                  id="scenario-description"
                  placeholder="Describe your scenario in natural language. For example: 'What happens if Drug A interacts with Drug B?' or 'Predict the effects of increasing temperature on reaction rate'..."
                  value={currentExperiment.scenarioDescription}
                  onChange={(e) =>
                    setCurrentExperiment({ ...currentExperiment, scenarioDescription: e.target.value })
                  }
                  rows={6}
                />
                <p className="text-xs text-muted-foreground">
                  The simulator will analyze this scenario using facts from your knowledge graph
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="input-data">
                  Additional Data (Optional)
                </Label>
                <Textarea
                  id="input-data"
                  placeholder='Add extra data not in your knowledge base (JSON or text). For example: {"temperature": 25, "pressure": 1.0}'
                  value={currentExperiment.inputData || ""}
                  onChange={(e) =>
                    setCurrentExperiment({ ...currentExperiment, inputData: e.target.value })
                  }
                  rows={4}
                  className="font-mono text-sm"
                />
                <p className="text-xs text-muted-foreground">
                  Optional: Provide additional data that is not already in your knowledge base. Leave empty if all data is in the knowledge graph.
                </p>
              </div>

              <Button
                onClick={handleRunExperiment}
                disabled={isRunning || !currentExperiment.name || !currentExperiment.scenarioDescription}
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
                              {experiment.config.scenarioDescription}
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
                      <div className="space-y-4">
                        <div>
                          <Label className="text-base font-semibold">Scenario Analysis</Label>
                          <p className="text-sm text-muted-foreground mt-1">
                            {selectedExperimentData.results.scenario_description}
                          </p>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <Label className="text-sm">Relevant Facts Found</Label>
                            <p className="text-2xl font-bold">{selectedExperimentData.results.relevant_facts_count || 0}</p>
                          </div>
                          <div>
                            <Label className="text-sm">Mentioned Entities</Label>
                            <p className="text-2xl font-bold">{selectedExperimentData.results.mentioned_entities?.length || 0}</p>
                          </div>
                        </div>

                        {selectedExperimentData.results.analysis && (
                          <div className="space-y-3">
                            {selectedExperimentData.results.analysis.conclusion && (
                              <Alert>
                                <AlertTitle>Conclusion</AlertTitle>
                                <AlertDescription>
                                  {selectedExperimentData.results.analysis.conclusion}
                                </AlertDescription>
                              </Alert>
                            )}

                            {selectedExperimentData.results.analysis.confidence !== undefined && (
                              <div>
                                <Label>Confidence Level</Label>
                                <div className="mt-1 flex items-center gap-2">
                                  <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                    <div
                                      className="h-full bg-primary transition-all"
                                      style={{ width: `${(selectedExperimentData.results.analysis.confidence * 100)}%` }}
                                    />
                                  </div>
                                  <span className="text-sm font-medium">
                                    {Math.round(selectedExperimentData.results.analysis.confidence * 100)}%
                                  </span>
                                </div>
                              </div>
                            )}

                            {selectedExperimentData.results.analysis.key_evidence && (
                              <div>
                                <Label>Key Evidence</Label>
                                <ul className="mt-2 space-y-1">
                                  {selectedExperimentData.results.analysis.key_evidence.map((evidence: string, idx: number) => (
                                    <li key={idx} className="text-sm text-muted-foreground">• {evidence}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {selectedExperimentData.results.analysis.predicted_outcomes && (
                              <div>
                                <Label>Predicted Outcomes</Label>
                                <ul className="mt-2 space-y-1">
                                  {selectedExperimentData.results.analysis.predicted_outcomes.map((outcome: string, idx: number) => (
                                    <li key={idx} className="text-sm">• {outcome}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {selectedExperimentData.results.analysis.potential_impacts && (
                              <div>
                                <Label>Potential Impacts</Label>
                                <ul className="mt-2 space-y-1">
                                  {selectedExperimentData.results.analysis.potential_impacts.map((impact: string, idx: number) => (
                                    <li key={idx} className="text-sm">• {impact}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {selectedExperimentData.results.analysis.validation_details && (
                              <div>
                                <Label>Validation Details</Label>
                                <ul className="mt-2 space-y-1">
                                  {selectedExperimentData.results.analysis.validation_details.map((detail: string, idx: number) => (
                                    <li key={idx} className="text-sm">• {detail}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {selectedExperimentData.results.mentioned_entities && selectedExperimentData.results.mentioned_entities.length > 0 && (
                              <div>
                                <Label>Key Entities</Label>
                                <div className="mt-2 flex flex-wrap gap-2">
                                  {selectedExperimentData.results.mentioned_entities.slice(0, 10).map((entity: string, idx: number) => (
                                    <Badge key={idx} variant="outline">{entity}</Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {selectedExperimentData.results.relevant_facts && selectedExperimentData.results.relevant_facts.length > 0 && (
                          <div>
                            <Label>Relevant Facts from Knowledge Graph</Label>
                            <div className="mt-2 space-y-2 max-h-64 overflow-y-auto">
                              {selectedExperimentData.results.relevant_facts.slice(0, 10).map((fact: any, idx: number) => (
                                <Card key={idx} className="p-3">
                                  <div className="text-sm">
                                    <span className="font-semibold">{fact.subject}</span>
                                    <span className="mx-2 text-muted-foreground">{fact.predicate}</span>
                                    <span className="font-semibold">{fact.object}</span>
                                    {fact.relevance_score && (
                                      <Badge variant="secondary" className="ml-2">
                                        Score: {fact.relevance_score}
                                      </Badge>
                                    )}
                                  </div>
                                </Card>
                              ))}
                              {selectedExperimentData.results.relevant_facts.length > 10 && (
                                <p className="text-xs text-muted-foreground text-center">
                                  ... and {selectedExperimentData.results.relevant_facts.length - 10} more facts
                                </p>
                              )}
                            </div>
                          </div>
                        )}

                        <details className="mt-4">
                          <summary className="cursor-pointer text-sm text-muted-foreground hover:text-foreground">
                            View Raw JSON Data
                          </summary>
                          <pre className="mt-2 p-4 bg-muted rounded-md text-xs overflow-auto max-h-64">
                            {JSON.stringify(selectedExperimentData.results, null, 2)}
                          </pre>
                        </details>
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
