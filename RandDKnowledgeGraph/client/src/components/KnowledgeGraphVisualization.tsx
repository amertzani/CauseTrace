import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { ZoomIn, ZoomOut, Maximize2, Search, RotateCcw, Edit3, Plus, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { GraphNode, GraphEdge } from "@shared/schema";
import * as d3Force from "d3-force";
import * as d3Drag from "d3-drag";
import * as d3Select from "d3-selection";

interface KnowledgeGraphVisualizationProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Override title (e.g. "Causal Graph" for causal page) */
  graphTitle?: string;
  onNodeClick?: (node: GraphNode) => void;
  onNodeEdit?: (node: GraphNode) => void;
  onNodeMove?: (nodeId: string, position: { x: number; y: number; z: number }) => void;
  onEdgeEdit?: (edge: GraphEdge) => void;
  onEdgeDelete?: (edge: GraphEdge) => void;
  onConnectionCreate?: (sourceId: string, targetId: string) => void;
}

interface TooltipData {
  x: number;
  y: number;
  content: React.ReactNode;
}

interface NodeWithPosition extends GraphNode {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

interface EdgeWithPositions extends GraphEdge {
  source: NodeWithPosition | string;
  target: NodeWithPosition | string;
}

// Enhanced color palette for different node types
const getNodeColor = (type: string, label: string): { fill: string; stroke: string; typeLabel: string } => {
  const lowerType = type.toLowerCase();
  const lowerLabel = label.toLowerCase();
  
  // Detect entity types from labels
  if (lowerLabel.match(/\b(person|dr\.|professor|researcher|scientist|engineer)\b/i)) {
    return { fill: "#60A5FA", stroke: "#3B82F6", typeLabel: "PERSON" };
  }
  if (lowerLabel.match(/\b(company|corporation|inc\.|ltd\.|organization|institution|university|college)\b/i)) {
    return { fill: "#34D399", stroke: "#10B981", typeLabel: "COMPANY" };
  }
  if (lowerLabel.match(/\b(project|task|work|deliverable|milestone)\b/i)) {
    return { fill: "#FBBF24", stroke: "#F59E0B", typeLabel: "PROJECT" };
  }
  if (lowerLabel.match(/\b(date|time|year|month|day|deadline|schedule)\b/i)) {
    return { fill: "#A78BFA", stroke: "#8B5CF6", typeLabel: "DATE" };
  }
  if (lowerLabel.match(/\b(location|place|city|country|address)\b/i)) {
    return { fill: "#F87171", stroke: "#EF4444", typeLabel: "LOCATION" };
  }
  if (lowerLabel.match(/\b(document|file|paper|report|article)\b/i)) {
    return { fill: "#FB7185", stroke: "#EC4899", typeLabel: "DOCUMENT" };
  }
  
  // Default colors based on type
  const colorMap: Record<string, { fill: string; stroke: string; typeLabel: string }> = {
    concept: { fill: "#94A3B8", stroke: "#64748B", typeLabel: "CONCEPT" },
    entity: { fill: "#60A5FA", stroke: "#3B82F6", typeLabel: "ENTITY" },
    process: { fill: "#34D399", stroke: "#10B981", typeLabel: "PROCESS" },
  };
  
  return colorMap[lowerType] || { fill: "#94A3B8", stroke: "#64748B", typeLabel: "CONCEPT" };
};

export function KnowledgeGraphVisualization({ 
  nodes, 
  edges,
  graphTitle = "Knowledge Graph Network",
  onNodeClick,
  onNodeEdit,
  onNodeMove,
  onEdgeEdit,
  onEdgeDelete,
  onConnectionCreate,
}: KnowledgeGraphVisualizationProps) {
  const [zoom, setZoom] = useState(1); // Start at normal zoom
  const [searchTerm, setSearchTerm] = useState("");
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [connectionMode, setConnectionMode] = useState(false);
  const [selectedNodeForConnection, setSelectedNodeForConnection] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<string>("all");
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [simulationTick, setSimulationTick] = useState(0);
  const [touchDistance, setTouchDistance] = useState<number | null>(null);
  const lastTouchDistanceRef = useRef<number | null>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);
  
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3Force.Simulation<NodeWithPosition, EdgeWithPositions> | null>(null);
  const nodesRef = useRef<NodeWithPosition[]>([]);
  const edgesRef = useRef<EdgeWithPositions[]>([]);
  const dragRef = useRef<d3Drag.DragBehavior<SVGGElement, NodeWithPosition> | null>(null);
  const nodeGroupsRef = useRef<Map<string, SVGGElement>>(new Map());

  // Initialize force simulation
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const width = containerRef.current?.clientWidth || 800;
    const height = containerRef.current?.clientHeight || 600;
    const centerX = width / 2;
    const centerY = height / 2;

    // Initialize nodes with positions - spread out for large graphs
    const nodesWithPos: NodeWithPosition[] = nodes.map((node, i) => {
      const angle = (i / nodes.length) * 2 * Math.PI;
      // Adaptive initial radius based on graph size
      let radius;
      if (nodes.length > 200) {
        radius = Math.min(300, Math.sqrt(nodes.length) * 15);
      } else if (nodes.length > 100) {
        radius = Math.min(250, Math.sqrt(nodes.length) * 12);
      } else {
        radius = Math.min(200, Math.sqrt(nodes.length) * 10);
      }
      return {
        ...node,
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
      };
    });

    // Create edges with node references
    const edgesWithPos: EdgeWithPositions[] = edges.map(edge => {
      const source = nodesWithPos.find(n => n.id === edge.source);
      const target = nodesWithPos.find(n => n.id === edge.target);
      return {
        ...edge,
        source: source || edge.source,
        target: target || edge.target,
      };
    }).filter(e => e.source && e.target) as EdgeWithPositions[];

    nodesRef.current = nodesWithPos;
    edgesRef.current = edgesWithPos;

    // Create force simulation with balanced parameters for large graphs
    const simulation = d3Force.forceSimulation<NodeWithPosition>(nodesWithPos)
      .force("link", d3Force.forceLink<NodeWithPosition, EdgeWithPositions>(edgesWithPos)
        .id((d: any) => d.id)
        .distance((d: any) => {
          // Adaptive link distances based on graph size
          if (nodes.length > 200) {
            return 80 + (d.source?.connections || 0) * 3;
          } else if (nodes.length > 100) {
            return 100 + (d.source?.connections || 0) * 4;
          } else {
            return 120 + (d.source?.connections || 0) * 5;
          }
        })
        .strength(0.3)) // Moderate link force
      .force("charge", d3Force.forceManyBody()
        .strength((d: any) => {
          // Balanced repulsion - stronger for larger graphs
          if (nodes.length > 200) {
            return -300;
          } else if (nodes.length > 100) {
            return -400;
          } else {
            return -500;
          }
        }))
      .force("center", d3Force.forceCenter(centerX, centerY).strength(0.1)) // Moderate center force
      .force("collision", d3Force.forceCollide()
        .radius((d: any) => {
          // Proper collision radius - match node size
          const baseRadius = 20;
          const connectionBonus = Math.min((d.connections || 0) * 3, 30);
          return baseRadius + connectionBonus + 5; // Extra padding
        })
        .strength(0.8)); // Strong collision to prevent overlap

    simulationRef.current = simulation;

    // Update positions on tick
    simulation.on("tick", () => {
      // Trigger re-render by updating state
      setSimulationTick(prev => prev + 1);
    });

    // Node drag will be attached in a separate effect after node groups are rendered
    dragRef.current = null;

    // Run simulation longer for better convergence, then stop and fix positions
    simulation.alphaDecay(0.05); // Faster decay for quicker stabilization
    simulation.alpha(1).restart(); // Start with full energy
    setTimeout(() => {
      // Fix all node positions after simulation completes
      nodesWithPos.forEach(node => {
        if (node.x !== undefined && node.y !== undefined) {
          node.fx = node.x;
          node.fy = node.y;
        }
      });
      simulation.stop();
    }, 5000); // Run longer for large graphs

    return () => {
      simulation.stop();
    };
  }, [nodes, edges]); // Removed pan, zoom, onNodeMove - simulation should not restart on pan/zoom

  // Attach d3-drag to node groups so nodes can be dragged (runs after nodes are rendered)
  useEffect(() => {
    if (simulationTick === 0 || !simulationRef.current || nodes.length === 0) return;
    const drag = d3Drag.drag<SVGGElement, NodeWithPosition>()
      .on("start", (event) => {
        event.sourceEvent?.stopPropagation?.(); // prevent canvas pan
        const node = event.subject;
        if (node != null && node.x != null && node.y != null) {
          node.fx = node.x;
          node.fy = node.y;
        }
      })
      .on("drag", (event) => {
        const node = event.subject;
        if (node != null) {
          node.fx = event.x;
          node.fy = event.y;
          setSimulationTick((t) => t + 1);
        }
      })
      .on("end", (event) => {
        const node = event.subject;
        if (node != null) {
          node.fx = event.x;
          node.fy = event.y;
          setSimulationTick((t) => t + 1);
          onNodeMove?.(node.id, { x: event.x, y: event.y, z: 0 });
        }
      });
    nodeGroupsRef.current.forEach((el, nodeId) => {
      const node = nodesRef.current.find((n) => n.id === nodeId);
      if (node) d3Select.select(el).datum(node).call(drag);
    });
    dragRef.current = drag;
  }, [simulationTick, nodes.length, onNodeMove]);

  // Filter nodes and edges
  const filteredData = useMemo(() => {
    let filteredNodes = nodes.filter(node =>
      node.label.toLowerCase().includes(searchTerm.toLowerCase())
    );

    if (filterType !== "all") {
      filteredNodes = filteredNodes.filter(node => {
        const colors = getNodeColor(node.type, node.label);
        return colors.typeLabel === filterType;
      });
    }

    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = edges.filter(edge =>
      filteredNodeIds.has(edge.source) && filteredNodeIds.has(edge.target)
    );

    return { nodes: filteredNodes, edges: filteredEdges };
  }, [nodes, edges, searchTerm, filterType, simulationTick]);

  // Get unique node types for filter
  const nodeTypes = useMemo(() => {
    const types = new Set<string>();
    nodes.forEach(node => {
      const colors = getNodeColor(node.type, node.label);
      types.add(colors.typeLabel);
    });
    return Array.from(types).sort();
  }, [nodes]);

  const handleZoomIn = useCallback(() => {
    setZoom(prev => Math.min(prev * 1.2, 3));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom(prev => Math.max(prev / 1.2, 0.3));
  }, []);

  const handleReset = useCallback(() => {
    // Auto-fit: Calculate optimal zoom to fit all nodes
    if (nodesRef.current.length > 0) {
      const width = containerRef.current?.clientWidth || 800;
      const height = containerRef.current?.clientHeight || 600;
      
      // Find bounding box of all nodes
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      nodesRef.current.forEach(node => {
        if (node.x !== undefined && node.y !== undefined) {
          minX = Math.min(minX, node.x);
          maxX = Math.max(maxX, node.x);
          minY = Math.min(minY, node.y);
          maxY = Math.max(maxY, node.y);
        }
      });
      
      const nodeWidth = maxX - minX;
      const nodeHeight = maxY - minY;
      const padding = 50;
      
      if (nodeWidth > 0 && nodeHeight > 0) {
        const scaleX = (width - padding * 2) / nodeWidth;
        const scaleY = (height - padding * 2) / nodeHeight;
        const optimalZoom = Math.min(scaleX, scaleY, 2); // Cap at 200%
        setZoom(optimalZoom);
        
        // Center the view
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        setPan({ x: -centerX, y: -centerY });
      } else {
        setZoom(1.2);
        setPan({ x: 0, y: 0 });
      }
      
      // Restart simulation if needed
      if (simulationRef.current) {
        const centerX = width / 2;
        const centerY = height / 2;
        
        nodesRef.current.forEach((node, i) => {
          const angle = (i / nodesRef.current.length) * 2 * Math.PI;
          const radius = Math.min(100, Math.sqrt(nodesRef.current.length) * 8);
          node.x = centerX + Math.cos(angle) * radius;
          node.y = centerY + Math.sin(angle) * radius;
          node.vx = 0;
          node.vy = 0;
        });
        
        simulationRef.current.alpha(1).restart();
        setTimeout(() => simulationRef.current?.stop(), 3000);
      }
    } else {
      setZoom(1.2);
      setPan({ x: 0, y: 0 });
    }
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Calculate distance between two touch points
  const getTouchDistance = (touch1: Touch, touch2: Touch): number => {
    const dx = touch1.clientX - touch2.clientX;
    const dy = touch1.clientY - touch2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
  };

  // Handle touch events for pinch-to-zoom
  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const distance = getTouchDistance(e.touches[0], e.touches[1]);
      lastTouchDistanceRef.current = distance;
      setTouchDistance(distance);
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 2 && lastTouchDistanceRef.current !== null) {
      e.preventDefault(); // Prevent scrolling
      const distance = getTouchDistance(e.touches[0], e.touches[1]);
      const scaleChange = distance / lastTouchDistanceRef.current;
      
      setZoom(prev => {
        const newZoom = prev * scaleChange;
        return Math.max(0.1, Math.min(5, newZoom)); // Limit zoom between 10% and 500%
      });
      
      lastTouchDistanceRef.current = distance;
    }
  };

  const handleTouchEnd = () => {
    lastTouchDistanceRef.current = null;
    setTouchDistance(null);
  };

  // Handle mouse wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => {
      const newZoom = prev * delta;
      return Math.max(0.1, Math.min(5, newZoom));
    });
  };

  const handleNodeClick = (node: GraphNode) => {
    if (connectionMode) {
      if (selectedNodeForConnection === null) {
        setSelectedNodeForConnection(node.id);
      } else if (selectedNodeForConnection !== node.id && onConnectionCreate) {
        onConnectionCreate(selectedNodeForConnection, node.id);
        setSelectedNodeForConnection(null);
        setConnectionMode(false);
      }
    } else {
      setSelectedNode(selectedNode === node.id ? null : node.id);
      onNodeClick?.(node);
    }
  };

  const toggleConnectionMode = () => {
    setConnectionMode(!connectionMode);
    setSelectedNodeForConnection(null);
  };

  const width = 800;
  const height = 600;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between gap-4 mb-4 flex-wrap">
        <h3 className="text-lg font-semibold">{graphTitle}</h3>
        <div className="flex items-center gap-2 flex-wrap">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9 w-48"
            />
          </div>
          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger className="w-40">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              {nodeTypes.map(type => (
                <SelectItem key={type} value={type}>{type}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant={connectionMode ? "default" : "outline"}
            size="sm"
            onClick={toggleConnectionMode}
          >
            <Plus className="h-4 w-4 mr-1" />
            {connectionMode ? "Cancel" : "Connect"}
          </Button>
          <Button variant="outline" size="icon" onClick={handleZoomOut}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={handleZoomIn}>
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div 
        ref={containerRef}
        className="border rounded-md bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 overflow-hidden relative" 
        style={{ height: "600px" }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <svg 
          ref={svgRef}
          width="100%" 
          height="100%" 
          className="absolute inset-0"
          style={{
            cursor: isDragging ? "grabbing" : connectionMode ? "crosshair" : "grab",
          }}
        >
          <defs>
            <marker 
              id="arrowhead" 
              markerWidth="10" 
              markerHeight="10" 
              refX="9" 
              refY="3" 
              orient="auto"
              markerUnits="strokeWidth"
            >
              <polygon points="0 0, 10 3, 0 6" fill="#64748B" opacity="0.6"/>
            </marker>
            <filter id="glow">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
            <filter id="shadow">
              <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.3"/>
            </filter>
          </defs>
          
          <g transform={`translate(${pan.x + width/2}, ${pan.y + height/2}) scale(${zoom})`}>
            {/* Render edges */}
            {filteredData.edges.map((edge) => {
              const source = nodesRef.current.find(n => n.id === edge.source) || 
                            (typeof edge.source === 'string' ? null : edge.source);
              const target = nodesRef.current.find(n => n.id === edge.target) || 
                            (typeof edge.target === 'string' ? null : edge.target);
              
              if (!source || !target) return null;
              const sx = source.x ?? width / 2;
              const sy = source.y ?? height / 2;
              const tx = target.x ?? width / 2;
              const ty = target.y ?? height / 2;
              
              const dx = tx - sx;
              const dy = ty - sy;
              const length = Math.sqrt(dx * dx + dy * dy);
              const angle = Math.atan2(dy, dx);
              
              // Match node radius calculation for edge positioning
              const sourceBaseRadius = 20;
              const sourceConnectionBonus = Math.min((source.connections || 0) * 3, 30);
              const sourceRadius = sourceBaseRadius + sourceConnectionBonus;
              
              const targetBaseRadius = 20;
              const targetConnectionBonus = Math.min((target.connections || 0) * 3, 30);
              const targetRadius = targetBaseRadius + targetConnectionBonus;
              
              const x1 = sx + Math.cos(angle) * sourceRadius;
              const y1 = sy + Math.sin(angle) * sourceRadius;
              const x2 = tx - Math.cos(angle) * targetRadius;
              const y2 = ty - Math.sin(angle) * targetRadius;
              
              const isHighlighted = hoveredNode === source.id || hoveredNode === target.id || hoveredEdge === edge.id;
              
              // Create invisible hit area for easier clicking/hovering
              const hitAreaWidth = Math.max(8, length * 0.1);
              
              return (
                <g 
                  key={edge.id}
                  onMouseEnter={(e) => {
                    setHoveredEdge(edge.id);
                    const rect = containerRef.current?.getBoundingClientRect();
                    if (rect) {
                      const midX = (x1 + x2) / 2;
                      const midY = (y1 + y2) / 2;
                      const screenX = (midX * zoom) + pan.x + (width / 2) + rect.left;
                      const screenY = (midY * zoom) + pan.y + (height / 2) + rect.top;
                      
                      const sourceLabel = typeof source === 'string' ? source : source.label;
                      const targetLabel = typeof target === 'string' ? target : target.label;
                      
                      setTooltip({
                        x: screenX,
                        y: screenY - 10,
                        content: (
                          <div className="bg-background border rounded-lg shadow-lg p-3 max-w-xs text-xs">
                            <div className="font-semibold mb-2 text-sm border-b pb-1">
                              {sourceLabel} → {targetLabel}
                            </div>
                            <div className="mb-1">
                              <span className="font-medium text-primary">Relationship:</span> {edge.label}
                            </div>
                            {edge.details && (
                              <div className="mb-1 mt-2 text-muted-foreground">
                                <span className="font-medium">Details:</span> {edge.details.length > 100 ? edge.details.substring(0, 100) + "..." : edge.details}
                              </div>
                            )}
                            {edge.sourceDocument && (
                              <div className="mb-1 text-muted-foreground">
                                <span className="font-medium">Source:</span> {edge.sourceDocument}
                              </div>
                            )}
                            {edge.uploadedAt && (
                              <div className="text-muted-foreground text-xs">
                                <span className="font-medium">Added:</span> {new Date(edge.uploadedAt).toLocaleDateString()}
                              </div>
                            )}
                            {edge.agent && (
                              <div className="mb-1 mt-2 pt-2 border-t">
                                <span className="font-medium text-xs">Processed by:</span>
                                <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                                  {edge.agent.replace(' Agent', '')}
                                </span>
                              </div>
                            )}
                            {edge.isInferred && (
                              <div className="mt-2 pt-2 border-t">
                                <span className="text-purple-400 font-medium text-xs">⚠️ Inferred Fact</span>
                                <p className="text-xs text-muted-foreground mt-1">This fact was inferred through transitive reasoning</p>
                              </div>
                            )}
                          </div>
                        ),
                      });
                    }
                  }}
                  onMouseLeave={() => {
                    setHoveredEdge(null);
                    if (!hoveredNode) {
                      setTooltip(null);
                    }
                  }}
                >
                  {/* Invisible hit area for easier interaction */}
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="transparent"
                    strokeWidth={hitAreaWidth}
                    style={{ cursor: "pointer" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onEdgeEdit?.(edge);
                    }}
                  />
                  
                  {/* Visible edge line - different style for inferred edges */}
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={edge.isInferred 
                      ? (hoveredEdge === edge.id ? "#8B5CF6" : "#A78BFA")  // Purple for inferred
                      : (hoveredEdge === edge.id ? "#3B82F6" : "#94A3B8")}  // Blue for original
                    strokeWidth={hoveredEdge === edge.id ? 3 : isHighlighted ? 2 : 1.5}
                    strokeDasharray={edge.isInferred ? "5,5" : "none"}  // Dashed for inferred
                    markerEnd={zoom > 0.5 ? "url(#arrowhead)" : undefined}
                    opacity={hoveredEdge === edge.id ? 0.9 : isHighlighted ? 0.7 : zoom > 0.6 ? 0.5 : 0.4}
                    style={{ cursor: "pointer", pointerEvents: "none" }}
                  />
                  
                  {/* Edge label - more visible */}
                  {length > 50 && zoom > 0.5 && (
                    <g>
                      {/* Background for text readability */}
                      <rect
                        x={(x1 + x2) / 2 - 25}
                        y={(y1 + y2) / 2 - 10}
                        width="50"
                        height="14"
                        fill="white"
                        fillOpacity="0.9"
                        rx="3"
                        style={{ pointerEvents: "none" }}
                      />
                      <text
                        x={(x1 + x2) / 2}
                        y={(y1 + y2) / 2 - 2}
                        fontSize={zoom > 1 ? "11" : "10"}
                        fill="#1E40AF"
                        textAnchor="middle"
                        style={{ pointerEvents: "none", userSelect: "none" }}
                        className="font-semibold"
                      >
                        {edge.label.length > 15 ? edge.label.substring(0, 15) + "..." : edge.label}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}
            
            {/* Render nodes */}
            {filteredData.nodes.map((node) => {
              const nodeWithPos = nodesRef.current.find(n => n.id === node.id);
              if (!nodeWithPos) return null;
              // Use current position or fallback
              const x = nodeWithPos.x ?? width / 2;
              const y = nodeWithPos.y ?? height / 2;
              
              const colors = getNodeColor(node.type, node.label);
              // Node size scales with connections - visible but not too large
              // Base: 20px, each connection adds 3px, max 50px
              const baseRadius = 20;
              const connectionBonus = Math.min((node.connections || 0) * 3, 30); // Cap at 30px bonus
              const radius = baseRadius + connectionBonus;
              const isHighlighted = hoveredNode === node.id;
              const isSelected = selectedNode === node.id;
              const isConnectionSelected = selectedNodeForConnection === node.id;
              
              return (
                <g
                  key={node.id}
                  className="node-group"
                  ref={(el) => {
                    if (el) {
                      nodeGroupsRef.current.set(node.id, el);
                    } else {
                      nodeGroupsRef.current.delete(node.id);
                    }
                  }}
                  transform={`translate(${x}, ${y})`}
                  style={{ cursor: connectionMode ? "crosshair" : "grab" }}
                  onMouseEnter={(e) => {
                    setHoveredNode(node.id);
                    const rect = containerRef.current?.getBoundingClientRect();
                    if (rect) {
                      const screenX = (x * zoom) + pan.x + (width / 2) + rect.left;
                      const screenY = (y * zoom) + pan.y + (height / 2) + rect.top;
                      
                      // Find all edges connected to this node for tooltip
                      const connectedEdges = filteredData.edges.filter(e => 
                        e.source === node.id || e.target === node.id
                      );
                      
                      setTooltip({
                        x: screenX,
                        y: screenY - radius - 20,
                        content: (
                          <div className="bg-background border rounded-lg shadow-lg p-3 max-w-xs text-xs">
                            <div className="font-semibold mb-2 text-sm border-b pb-1 flex items-center gap-2">
                              <div 
                                className="h-3 w-3 rounded-full border-2" 
                                style={{ 
                                  backgroundColor: colors.fill, 
                                  borderColor: colors.stroke 
                                }} 
                              />
                              {colors.typeLabel}: {node.label}
                            </div>
                            <div className="mb-1">
                              <span className="font-medium text-primary">Connections:</span> {node.connections}
                            </div>
                            {connectedEdges.length > 0 && (
                              <div className="mt-2">
                                <div className="font-medium mb-1">Relationships:</div>
                                <div className="space-y-1 max-h-32 overflow-y-auto">
                                  {connectedEdges.slice(0, 5).map(e => {
                                    const otherNodeId = e.source === node.id ? e.target : e.source;
                                    const otherNode = filteredData.nodes.find(n => n.id === otherNodeId);
                                    const direction = e.source === node.id ? "→" : "←";
                                    return (
                                      <div key={e.id} className="text-muted-foreground">
                                        {direction} <span className="font-medium">{e.label}</span> {direction === "→" ? otherNode?.label : otherNode?.label}
                                      </div>
                                    );
                                  })}
                                  {connectedEdges.length > 5 && (
                                    <div className="text-muted-foreground italic">
                                      +{connectedEdges.length - 5} more...
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        ),
                      });
                    }
                  }}
                  onMouseLeave={() => {
                    setHoveredNode(null);
                    if (!hoveredEdge) {
                      setTooltip(null);
                    }
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleNodeClick(node);
                  }}
                >
                  {/* Shadow */}
                  <circle
                    r={radius}
                    fill="black"
                    opacity="0.1"
                    transform="translate(0, 2)"
                  />
                  
                  {/* Node circle */}
                  <circle
                    r={radius}
                    fill={colors.fill}
                    fillOpacity={isConnectionSelected ? 0.5 : isSelected ? 0.9 : isHighlighted ? 0.8 : 0.7}
                    stroke={isSelected ? "#1E40AF" : colors.stroke}
                    strokeWidth={isSelected ? 3 : isHighlighted ? 2.5 : 2}
                    filter={isHighlighted ? "url(#glow)" : "url(#shadow)"}
                    className="transition-all"
                  />
                  
                  {/* Node label - Type (only show if zoomed in enough) */}
                  {zoom > 0.8 && (
                    <text
                      y={-radius - 6}
                      textAnchor="middle"
                      fontSize="9"
                      fill={colors.stroke}
                      fontWeight="600"
                      style={{ pointerEvents: "none", userSelect: "none" }}
                      className="uppercase tracking-wide"
                    >
                      {colors.typeLabel}
                    </text>
                  )}
                  
                  {/* Node label - Name (only show if zoomed in enough) */}
                  {zoom > 0.5 && (
                    <text
                      y={radius / 2}
                      textAnchor="middle"
                      fontSize={Math.max(9, Math.min(12, 11 - node.label.length / 15))}
                      fill="white"
                      fontWeight="500"
                      style={{ pointerEvents: "none", userSelect: "none" }}
                      className="font-semibold"
                    >
                      {zoom > 1 ? node.label : (node.label.length > 12 ? node.label.substring(0, 12) + "..." : node.label)}
                    </text>
                  )}
                  
                  {/* Edit button on hover */}
                  {isHighlighted && !connectionMode && (
                    <g
                      onClick={(e) => {
                        e.stopPropagation();
                        onNodeEdit?.(node);
                      }}
                      style={{ cursor: "pointer" }}
                      transform="translate(0, 0)"
                    >
                      <circle
                        r={12}
                        fill="#1E40AF"
                        className="transition-all"
                      />
                      <g transform="translate(-6, -6)">
                        <Edit3
                          width={12}
                          height={12}
                          stroke="white"
                          strokeWidth="2"
                        />
                      </g>
                    </g>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
        
        {/* Tooltip overlay - positioned absolutely */}
        {tooltip && (
          <div
            className="absolute z-50 pointer-events-none"
            style={{
              left: `${tooltip.x}px`,
              top: `${tooltip.y}px`,
              transform: 'translate(-50%, -100%)',
            }}
          >
            {tooltip.content}
          </div>
        )}

        <div className="absolute bottom-4 right-4 flex flex-col gap-2 pointer-events-none">
          <Badge variant="secondary" className="text-xs">Zoom: {Math.round(zoom * 100)}%</Badge>
          <Badge variant="secondary" className="text-xs">Nodes: {filteredData.nodes.length}</Badge>
          <Badge variant="secondary" className="text-xs">Edges: {filteredData.edges.length}</Badge>
          {connectionMode && (
            <Badge variant="default" className="text-xs">
              {selectedNodeForConnection ? "Click target node" : "Click source node"}
            </Badge>
          )}
        </div>
        
        <div className="absolute top-4 left-4 text-xs text-muted-foreground bg-background/95 backdrop-blur-sm p-3 rounded-md border max-w-xs shadow-lg">
          <p className="font-semibold mb-1">💡 Navigation:</p>
          <ul className="space-y-1">
            <li>• <strong>Drag canvas</strong> to pan</li>
            <li>• <strong>Drag nodes</strong> to reposition</li>
            <li>• <strong>Pinch/scroll</strong> to zoom</li>
            <li>• <strong>Click nodes</strong> to select</li>
            <li>• <strong>Click "Connect"</strong> to link nodes</li>
            <li>• <strong>Hover & click edit</strong> to modify</li>
            <li>• <strong>Node size</strong> = connectivity</li>
          </ul>
        </div>
      </div>

      <div className="flex items-center gap-4 mt-4 flex-wrap">
        {nodeTypes.map(type => {
          const sampleNode = nodes.find(n => {
            const colors = getNodeColor(n.type, n.label);
            return colors.typeLabel === type;
          });
          if (!sampleNode) return null;
          const colors = getNodeColor(sampleNode.type, sampleNode.label);
          return (
            <div key={type} className="flex items-center gap-2">
              <div 
                className="h-4 w-4 rounded-full border-2" 
                style={{ 
                  backgroundColor: colors.fill, 
                  borderColor: colors.stroke 
                }} 
              />
              <span className="text-xs text-muted-foreground font-medium">{type}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
