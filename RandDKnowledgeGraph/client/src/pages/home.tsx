import { useState, useEffect } from "react";
import { ArrowRight, Upload, Database, Network, MessageSquare } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatsCards } from "@/components/StatsCards";
import { useLocation } from "wouter";
import { hfApi } from "@/lib/api-client";

export default function HomePage() {
  const [, setLocation] = useLocation();
  const [stats, setStats] = useState({
    documentsCount: 0,
    factsCount: 0,
    nodesCount: 0,
    connectionsCount: 0,
  });

  useEffect(() => {
    let cancelled = false;
    hfApi.getDashboardStats().then((res) => {
      if (cancelled || !res.success || !res.data) return;
      setStats({
        documentsCount: res.data.documents_count ?? 0,
        factsCount: res.data.facts_count ?? 0,
        nodesCount: res.data.nodes_count ?? 0,
        connectionsCount: res.data.connections_count ?? 0,
      });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const features = [
    {
      title: "Upload Documents",
      description: "Import research papers, datasets, and documentation",
      icon: Upload,
      href: "/upload",
    },
    {
      title: "Knowledge Base",
      description: "View and manage extracted facts and relationships",
      icon: Database,
      href: "/knowledge-base",
    },
    {
      title: "Knowledge Graph",
      description: "Visualize and explore your research network",
      icon: Network,
      href: "/graph",
    },
    {
      title: "Research Assistant",
      description: "Chat with AI about your research data",
      icon: MessageSquare,
      href: "/chat",
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-semibold mb-3">Welcome to Research Brain</h1>
        <p className="text-lg text-muted-foreground">
          Your intelligent R&D knowledge management system
        </p>
      </div>

      <StatsCards
        documentsCount={stats.documentsCount}
        factsCount={stats.factsCount}
        nodesCount={stats.nodesCount}
        connectionsCount={stats.connectionsCount}
      />

      <div>
        <h2 className="text-2xl font-semibold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {features.map((feature) => (
            <Card
              key={feature.title}
              className="p-6 hover-elevate cursor-pointer"
              onClick={() => setLocation(feature.href)}
              data-testid={`card-${feature.href.slice(1)}`}
            >
              <div className="flex items-start gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-md bg-primary text-primary-foreground">
                  <feature.icon className="h-6 w-6" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    {feature.description}
                  </p>
                  <Button variant="ghost" size="sm" className="gap-2">
                    Get Started
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      <Card className="p-6 bg-primary/5 border-primary/20">
        <h3 className="text-lg font-semibold mb-2">Getting Started</h3>
        <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
          <li>Upload your research documents (PDF, DOCX, CSV, TXT, PPTX)</li>
          <li>Process documents to extract knowledge facts automatically</li>
          <li>Explore the knowledge graph to visualize relationships</li>
          <li>Chat with the AI assistant to query your research data</li>
          <li>Export your knowledge base for backup or sharing</li>
        </ol>
      </Card>
    </div>
  );
}
