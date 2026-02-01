import {
  Database,
  Upload,
  Network,
  MessageSquare,
  FileText,
  Download,
  Settings,
  FlaskConical,
  GitBranch,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { useLocation } from "wouter";
import logoC from "@/assets/logo_C.jpeg";

const menuItems = [
  {
    title: "Upload Documents",
    url: "/upload",
    icon: Upload,
  },
  {
    title: "Knowledge Base",
    url: "/knowledge-base",
    icon: Database,
  },
  {
    title: "Knowledge Graph",
    url: "/graph",
    icon: Network,
  },
  {
    title: "Research Assistant",
    url: "/chat",
    icon: MessageSquare,
  },
  {
    title: "Documents",
    url: "/documents",
    icon: FileText,
  },
  {
    title: "Simulator",
    url: "/simulator",
    icon: FlaskConical,
  },
  {
    title: "Causal Graph",
    url: "/causal-graph",
    icon: GitBranch,
  },
];

const managementItems = [
  {
    title: "Import/Export",
    url: "/import-export",
    icon: Download,
  },
];

export function AppSidebar() {
  const [location] = useLocation();

  return (
    <Sidebar>
      <SidebarHeader className="p-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground overflow-hidden">
            <img 
              src={logoC}
              alt="Research Brain Logo" 
              className="h-full w-full object-cover"
            />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Research Brain</h2>
            <p className="text-xs text-muted-foreground">Knowledge Management</p>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Main Functions</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild isActive={location === item.url}>
                    <a href={item.url} data-testid={`link-${item.url.slice(1)}`}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Management</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {managementItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild isActive={location === item.url}>
                    <a href={item.url} data-testid={`link-${item.url.slice(1)}`}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="p-4">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild>
              <a href="/settings" data-testid="link-settings">
                <Settings className="h-4 w-4" />
                <span>Settings</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <div className="mt-4 pt-4 border-t flex justify-center">
          <img 
            src="/logo_all.jpeg" 
            alt="Logo" 
            className="h-8 w-auto opacity-60 hover:opacity-100 transition-opacity"
          />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
