import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { ThemeToggle } from "@/components/ThemeToggle";
import { KnowledgeStoreProvider } from "@/lib/knowledge-store";
import HomePage from "@/pages/home";
import UploadPage from "@/pages/upload";
import KnowledgeBasePage from "@/pages/knowledge-base";
import GraphPage from "@/pages/graph";
import ChatPage from "@/pages/chat";
import DocumentsPage from "@/pages/documents";
import ImportExportPage from "@/pages/import-export";
import SimulatorPage from "@/pages/simulator";
import CausalGraphPage from "@/pages/causal-graph";
import NotFound from "@/pages/not-found";

function Router() {
  return (
    <Switch>
      <Route path="/" component={HomePage} />
      <Route path="/upload" component={UploadPage} />
      <Route path="/knowledge-base" component={KnowledgeBasePage} />
      <Route path="/graph" component={GraphPage} />
      <Route path="/chat" component={ChatPage} />
      <Route path="/documents" component={DocumentsPage} />
      <Route path="/import-export" component={ImportExportPage} />
      <Route path="/simulator" component={SimulatorPage} />
      <Route path="/causal-graph" component={CausalGraphPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const style = {
    "--sidebar-width": "20rem",
    "--sidebar-width-icon": "4rem",
  };

  return (
    <QueryClientProvider client={queryClient}>
      <KnowledgeStoreProvider>
        <TooltipProvider>
          <SidebarProvider style={style as React.CSSProperties}>
            <div className="flex h-screen w-full">
              <AppSidebar />
              <div className="flex flex-col flex-1 overflow-hidden">
                <header className="flex items-center justify-between p-4 border-b bg-background">
                  <SidebarTrigger data-testid="button-sidebar-toggle" />
                  <ThemeToggle />
                </header>
                <main className="flex-1 overflow-auto p-8">
                  <Router />
                </main>
              </div>
            </div>
          </SidebarProvider>
          <Toaster />
        </TooltipProvider>
      </KnowledgeStoreProvider>
    </QueryClientProvider>
  );
}

export default App;
