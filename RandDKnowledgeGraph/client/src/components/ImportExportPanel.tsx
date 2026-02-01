import { useState } from "react";
import { Download, Upload, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface ImportExportPanelProps {
  onExport: (includeInferred: boolean, minConfidence: number) => void;
  onExportCausalGraph?: () => void;
  onImport: (file: File) => void;
  metadata?: {
    version: string;
    totalFacts: number;
    lastUpdated: string;
  };
}

export function ImportExportPanel({ onExport, onExportCausalGraph, onImport, metadata }: ImportExportPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [includeInferred, setIncludeInferred] = useState(true);
  const [minConfidence, setMinConfidence] = useState([0.0]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleImport = () => {
    if (selectedFile) {
      onImport(selectedFile);
      setSelectedFile(null);
    }
  };

  const handleExportClick = () => {
    setExportDialogOpen(true);
  };

  const handleExportConfirm = () => {
    onExport(includeInferred, minConfidence[0]);
    setExportDialogOpen(false);
  };

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Export Knowledge Base</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Download a backup of your complete knowledge base as a JSON file.
        </p>
        {metadata && (
          <div className="flex flex-wrap gap-2 mb-4">
            <Badge variant="secondary">Version {metadata.version}</Badge>
            <Badge variant="secondary">{metadata.totalFacts} facts</Badge>
            <Badge variant="secondary">
              Updated {new Date(metadata.lastUpdated).toLocaleDateString()}
            </Badge>
          </div>
        )}
        <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
          <DialogTrigger asChild>
            <Button onClick={handleExportClick} data-testid="button-export">
              <Download className="h-4 w-4 mr-2" />
              Download Knowledge Base
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Export Options</DialogTitle>
              <DialogDescription>
                Choose which facts to include in the export.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-6 py-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="export-inferred" className="text-base">
                    Include Inferred Facts
                  </Label>
                  <p className="text-sm text-muted-foreground">
                    Export facts that were inferred through transitive reasoning
                  </p>
                </div>
                <Switch
                  id="export-inferred"
                  checked={includeInferred}
                  onCheckedChange={setIncludeInferred}
                />
              </div>
              
              <div className="space-y-3">
                <div className="space-y-0.5">
                  <Label className="text-base">
                    Minimum Confidence: {minConfidence[0].toFixed(1)}
                  </Label>
                  <p className="text-sm text-muted-foreground">
                    Only export facts with confidence above this threshold
                  </p>
                </div>
                <Slider
                  value={minConfidence}
                  onValueChange={setMinConfidence}
                  min={0}
                  max={1}
                  step={0.1}
                  className="w-full"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setExportDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleExportConfirm}>
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </Card>

      {onExportCausalGraph && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Export Causal Graph</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Download causal relationships (KB + all data-driven datasets) as JSON for backup or sharing.
          </p>
          <Button onClick={onExportCausalGraph} variant="outline" data-testid="button-export-causal">
            <Download className="h-4 w-4 mr-2" />
            Download Causal Graph
          </Button>
        </Card>
      )}

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Import Knowledge Base</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Upload a previously exported knowledge base JSON file to restore or merge data.
        </p>
        <Alert className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Importing will merge with your existing knowledge base. Duplicate facts will be skipped.
          </AlertDescription>
        </Alert>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="import-file">Select JSON File</Label>
            <Input
              id="import-file"
              type="file"
              accept=".json"
              onChange={handleFileSelect}
              data-testid="input-import-file"
            />
          </div>
          {selectedFile && (
            <p className="text-sm text-muted-foreground">
              Selected: {selectedFile.name}
            </p>
          )}
          <Button
            onClick={handleImport}
            disabled={!selectedFile}
            data-testid="button-import"
          >
            <Upload className="h-4 w-4 mr-2" />
            Import Knowledge Base
          </Button>
        </div>
      </Card>
    </div>
  );
}
