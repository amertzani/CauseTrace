import { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';
import type { Fact, GraphNode, GraphEdge, InsertFact } from '@shared/schema';
import { hfApi } from './api-client';

interface KnowledgeStoreContextType {
  facts: Fact[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  addFact: (fact: InsertFact) => Promise<void>;
  updateFact: (id: string, updates: Partial<Fact>) => Promise<void>;
  deleteFact: (id: string) => Promise<void>;
  updateNode: (id: string, updates: Partial<GraphNode>) => Promise<void>;
  updateEdge: (id: string, updates: Partial<GraphEdge>) => Promise<void>;
  deleteEdge: (id: string) => Promise<void>;
  addEdge: (sourceId: string, targetId: string, label?: string) => Promise<void>;
  updateNodePosition: (nodeId: string, position: { x: number; y: number; z: number }) => void;
  refreshFacts?: () => Promise<any[] | void>;
}

const KnowledgeStoreContext = createContext<KnowledgeStoreContextType | null>(null);

export function useKnowledgeStore() {
  const context = useContext(KnowledgeStoreContext);
  if (!context) {
    throw new Error('useKnowledgeStore must be used within KnowledgeStoreProvider');
  }
  return context;
}

interface KnowledgeStoreProviderProps {
  children: ReactNode;
}

export function KnowledgeStoreProvider({ children }: KnowledgeStoreProviderProps) {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);

  const [nodePositions] = useState<Map<string, { x: number; y: number; z: number }>>(new Map());

  // Fetch initial facts from backend on mount
  useEffect(() => {
    async function loadFacts() {
      console.log('KnowledgeStoreProvider: Loading facts on mount...');
      const result = await hfApi.getKnowledgeBase();
      if (result.success && result.data?.facts) {
        const backendFacts = result.data.facts.map((f: any) => ({
          ...f,
          id: String(f.id), // Ensure ID is string
        }));
        console.log(`KnowledgeStoreProvider: Loaded ${backendFacts.length} facts on mount`);
        setFacts(backendFacts);
        
        // Rebuild graph from facts
        rebuildGraphFromFacts(backendFacts);
      } else {
        console.error('KnowledgeStoreProvider: Failed to load facts:', result);
      }
    }
    loadFacts();
  }, []); // Empty deps - only run on mount

  const rebuildGraphFromFacts = useCallback((factsList: Fact[]) => {
    const nodeMap = new Map<string, GraphNode>();
    const edgesList: GraphEdge[] = [];
    
    factsList.forEach((fact) => {
      // Create or get source node
      if (!nodeMap.has(fact.subject)) {
        nodeMap.set(fact.subject, {
          id: `node_${fact.subject.toLowerCase().replace(/\s+/g, '_')}`,
          label: fact.subject,
          type: "concept",
          connections: 0,
        });
      }
      
      // Create or get target node
      if (!nodeMap.has(fact.object)) {
        nodeMap.set(fact.object, {
          id: `node_${fact.object.toLowerCase().replace(/\s+/g, '_')}`,
          label: fact.object,
          type: "concept",
          connections: 0,
        });
      }
      
      // Create edge with full fact information
      const sourceNode = nodeMap.get(fact.subject)!;
      const targetNode = nodeMap.get(fact.object)!;
      edgesList.push({
        id: `edge_${fact.id}`,
        source: sourceNode.id,
        target: targetNode.id,
        label: fact.predicate,
        details: fact.details || undefined,
        sourceDocument: fact.sourceDocument || undefined,
        uploadedAt: fact.uploadedAt || undefined,
        isInferred: fact.isInferred || false,  // Include inferred status
        agent: fact.agent || undefined,  // Include agent information
      });
    });
    
    setNodes(Array.from(nodeMap.values()));
    setEdges(edgesList);
    recalculateConnections(edgesList);
  }, []);

  // Expose a refresh function
  const refreshFacts = useCallback(async () => {
    console.log('🔄 refreshFacts: Starting refresh from backend...');
    try {
      const result = await hfApi.getKnowledgeBase();
      console.log('🔄 refreshFacts: API response success:', result.success);
      console.log('🔄 refreshFacts: API response data keys:', result.data ? Object.keys(result.data) : 'no data');
      
      // Check multiple possible response formats
      const facts = result.data?.facts || result.data || [];
      console.log('🔄 refreshFacts: Extracted facts array length:', Array.isArray(facts) ? facts.length : 'not an array');
      
      if (result.success && Array.isArray(facts) && facts.length > 0) {
        const backendFacts = facts.map((f: any) => ({
          ...f,
          id: String(f.id || Date.now() + Math.random()), // Ensure unique ID
          details: f.details || null, // Preserve null/undefined details
        }));
        console.log(`✅ refreshFacts: Loaded ${backendFacts.length} facts from backend`);
        console.log('🔄 refreshFacts: First fact:', backendFacts[0]);
        console.log('🔄 refreshFacts: First fact details:', backendFacts[0]?.details);
        console.log('🔄 refreshFacts: Last fact:', backendFacts[backendFacts.length - 1]);
        setFacts(backendFacts);
        rebuildGraphFromFacts(backendFacts);
        console.log(`✅ refreshFacts: Updated state with ${backendFacts.length} facts and rebuilt graph`);
        return backendFacts;
      } else if (result.success && Array.isArray(facts)) {
        // Empty array is OK
        console.log(`✅ refreshFacts: Loaded 0 facts (empty database)`);
        setFacts([]);
        rebuildGraphFromFacts([]);
        return [];
      } else {
        console.error('❌ refreshFacts: Failed to refresh facts:', result);
        console.error('❌ refreshFacts: Facts type:', typeof facts, 'Is array:', Array.isArray(facts));
        console.error('❌ refreshFacts: Full response:', JSON.stringify(result, null, 2));
        throw new Error('Failed to refresh facts');
      }
    } catch (error) {
      console.error('❌ refreshFacts: Exception:', error);
      throw error;
    }
  }, [rebuildGraphFromFacts]);

  const recalculateConnections = useCallback((currentEdges: GraphEdge[]) => {
    const connectionMap = new Map<string, number>();
    currentEdges.forEach(edge => {
      connectionMap.set(edge.source, (connectionMap.get(edge.source) || 0) + 1);
      connectionMap.set(edge.target, (connectionMap.get(edge.target) || 0) + 1);
    });
    
    setNodes(prev => prev.map(node => ({
      ...node,
      connections: connectionMap.get(node.id) || 0,
    })));
  }, []);

  const findOrCreateNode = useCallback((label: string, currentNodes: GraphNode[], type: string = "concept"): { nodes: GraphNode[], nodeId: string } => {
    const existing = currentNodes.find(n => n.label.toLowerCase() === label.toLowerCase());
    if (existing) {
      return { nodes: currentNodes, nodeId: existing.id };
    }
    
    const newNodeId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const newNode: GraphNode = {
      id: newNodeId,
      label: label,
      type: type,
      connections: 0,
    };
    
    return { nodes: [...currentNodes, newNode], nodeId: newNodeId };
  }, []);

  const addFact = useCallback(async (factData: InsertFact) => {
    console.log('Adding fact:', factData);
    // Call backend API
    const result = await hfApi.createFact(factData);
    
    // Handle duplicate fact response
    if (result.success && result.data?.status === 'duplicate') {
      console.log('⚠️ Duplicate fact detected:', result.data.message);
      // Return a rejection so the UI can show an error message
      throw new Error(result.data.message || 'Fact already exists in knowledge graph');
    }
    
    if (result.success && result.data?.fact) {
      const newFact: Fact = {
        ...result.data.fact,
        id: String(result.data.fact.id), // Ensure ID is string
      };
      console.log('Fact created successfully:', newFact);
      
      // Immediately add to local state for instant UI update
      setFacts(prev => {
        // Check if fact already exists (avoid duplicates)
        const exists = prev.some(f => 
          f.subject === newFact.subject && 
          f.predicate === newFact.predicate && 
          f.object === newFact.object
        );
        if (exists) {
          console.log('Fact already exists, skipping duplicate');
          return prev;
        }
        const updatedFacts = [...prev, newFact];
        // Rebuild graph from all facts including the new one
        rebuildGraphFromFacts(updatedFacts);
        console.log(`Added fact to local state. Total facts: ${updatedFacts.length}`);
        return updatedFacts;
      });
      
      // Also reload from backend in background to ensure consistency
      // This ensures we have the exact fact ID and any backend processing
      setTimeout(async () => {
        const refreshResult = await hfApi.getKnowledgeBase();
        if (refreshResult.success && refreshResult.data?.facts) {
          const allFacts = refreshResult.data.facts.map((f: any) => ({
            ...f,
            id: String(f.id),
          }));
          console.log(`Background refresh: Loaded ${allFacts.length} facts from backend`);
          // Only update if count changed (backend might have processed differently)
          if (allFacts.length !== facts.length) {
            setFacts(allFacts);
            rebuildGraphFromFacts(allFacts);
          }
        }
      }, 100);
    } else {
      console.error('Failed to create fact:', result);
      // If API call failed but we have the fact data, add it optimistically
      const optimisticFact: Fact = {
        id: String(Date.now()),
        subject: factData.subject,
        predicate: factData.predicate,
        object: factData.object,
        source: factData.source || "manual"
      };
      setFacts(prev => {
        const updatedFacts = [...prev, optimisticFact];
        rebuildGraphFromFacts(updatedFacts);
        return updatedFacts;
      });
    }
  }, [rebuildGraphFromFacts, facts.length]);

  const updateFact = useCallback(async (id: string, updates: Partial<Fact>) => {
    const oldFact = facts.find(f => f.id === id);
    if (!oldFact) {
      console.error('Fact not found:', id);
      throw new Error('Fact not found');
    }
    
    // Calculate new values
    const newSubject = updates.subject || oldFact.subject;
    const newObject = updates.object || oldFact.object;
    const newPredicate = updates.predicate || oldFact.predicate;
    const newDetails = updates.details !== undefined ? updates.details : oldFact.details;
    
    // Check if anything actually changed
    const hasChanges = 
      newSubject !== oldFact.subject || 
      newObject !== oldFact.object || 
      newPredicate !== oldFact.predicate ||
      newDetails !== oldFact.details;
    
    if (!hasChanges) {
      console.log('No changes to fact, skipping update');
      return;
    }
    
    try {
      // Delete the old fact from backend
      const oldFactData = {
        subject: oldFact.subject,
        predicate: oldFact.predicate,
        object: oldFact.object
      };
      console.log('Deleting old fact from backend:', oldFactData);
      await hfApi.deleteFact(id, oldFactData);
      
      // Create the new fact with updated values
      const newFactData = {
        subject: newSubject,
        predicate: newPredicate,
        object: newObject,
        source: oldFact.source || "manual",
        details: newDetails || undefined
      };
      console.log('Creating new fact in backend:', newFactData);
      await addFact(newFactData);
      
      // Refresh facts to rebuild graph with updated fact
      await refreshFacts();
      
      console.log('Fact updated successfully in backend and graph');
    } catch (error) {
      console.error('Failed to update fact:', error);
      throw error;
    }
  }, [facts, addFact, refreshFacts]);

  const deleteFact = useCallback(async (id: string) => {
    const fact = facts.find(f => f.id === id);
    if (!fact) return;
    
    // Call backend API with fact data for proper deletion
    const factData = {
      subject: fact.subject,
      predicate: fact.predicate,
      object: fact.object
    };
    
    const result = await hfApi.deleteFact(id, factData);
    
    if (result.success) {
      // Remove from local facts
      setFacts(prev => prev.filter(f => f.id !== id));
      
      // Remove corresponding edge from graph
      const sourceNode = nodes.find(n => n.label === fact.subject);
      const targetNode = nodes.find(n => n.label === fact.object);
      
      if (sourceNode && targetNode) {
        setEdges(prev => {
          const updated = prev.filter(e => !(
            e.source === sourceNode.id && 
            e.target === targetNode.id && 
            e.label === fact.predicate
          ));
          recalculateConnections(updated);
          return updated;
        });
      }
      
      // Refresh facts from backend to ensure sync
      await refreshFacts();
    } else {
      console.error('Failed to delete fact:', result.error);
      throw new Error(result.error || 'Failed to delete fact');
    }
  }, [facts, nodes, recalculateConnections, refreshFacts]);

  const updateNode = useCallback(async (id: string, updates: Partial<GraphNode>) => {
    const oldNode = nodes.find(n => n.id === id);
    if (!oldNode) return;
    
    // Update local state immediately
    setNodes(prev => prev.map(n => n.id === id ? { ...n, ...updates } : n));
    
    if (updates.label) {
      // Find all facts that reference this node (as subject or object)
      const factsToUpdate = facts.filter(f => 
        f.subject === oldNode.label || f.object === oldNode.label
      );
      
      // Update each fact in the backend
      for (const fact of factsToUpdate) {
        try {
          const updatedFact = {
            ...fact,
            subject: fact.subject === oldNode.label ? updates.label! : fact.subject,
            object: fact.object === oldNode.label ? updates.label! : fact.object,
          };
          
          // Delete old fact and create new one (since backend doesn't have direct update)
          const factData = {
            subject: fact.subject,
            predicate: fact.predicate,
            object: fact.object
          };
          await hfApi.deleteFact(fact.id, factData);
          
          // Create new fact with updated label
          await hfApi.createFact({
            subject: updatedFact.subject,
            predicate: updatedFact.predicate,
            object: updatedFact.object,
            details: updatedFact.details,
            sourceDocument: updatedFact.sourceDocument,
            uploadedAt: updatedFact.uploadedAt,
          });
        } catch (error) {
          console.error(`Failed to update fact for node ${id}:`, error);
          // Continue with other facts even if one fails
        }
      }
      
      // Update local facts state
      setFacts(prev => prev.map(f => {
        let updated = { ...f };
        if (f.subject === oldNode.label) {
          updated.subject = updates.label || f.subject;
        }
        if (f.object === oldNode.label) {
          updated.object = updates.label || f.object;
        }
        return updated;
      }));
      
      // Refresh facts to ensure sync
      await refreshFacts();
    }
  }, [nodes, facts, refreshFacts]);

  const updateEdge = useCallback(async (id: string, updates: Partial<GraphEdge>) => {
    const oldEdge = edges.find(e => e.id === id);
    if (!oldEdge) return;
    
      const sourceNode = nodes.find(n => n.id === oldEdge.source);
      const targetNode = nodes.find(n => n.id === oldEdge.target);
      
    if (!sourceNode || !targetNode) {
      console.error('Cannot update edge: source or target node not found');
      return;
    }
    
    // Find the corresponding fact
    const factToUpdate = facts.find(f => 
      f.subject === sourceNode.label && 
      f.object === targetNode.label && 
      f.predicate === oldEdge.label
    );
    
    if (updates.label && factToUpdate) {
      // Update the fact in the backend
      try {
        // Delete the old fact
        const factData = {
          subject: factToUpdate.subject,
          predicate: factToUpdate.predicate,
          object: factToUpdate.object
        };
        await hfApi.deleteFact(factToUpdate.id, factData);
        
        // Add the new fact with updated predicate
        const newFactData = {
          subject: sourceNode.label,
          predicate: updates.label,
          object: targetNode.label,
        };
        await addFact(newFactData);
        
        // Refresh facts to ensure sync
        await refreshFacts();
      } catch (error) {
        console.error('Failed to update edge fact:', error);
        throw error;
      }
    } else {
      // Just update local state if no fact found (shouldn't happen)
      setEdges(prev => prev.map(e => e.id === id ? { ...e, ...updates } : e));
    }
  }, [edges, nodes, facts, addFact, refreshFacts]);

  const deleteEdge = useCallback(async (id: string) => {
    const edge = edges.find(e => e.id === id);
    if (!edge) return;
    
    const sourceNode = nodes.find(n => n.id === edge.source);
    const targetNode = nodes.find(n => n.id === edge.target);
    
    if (!sourceNode || !targetNode) {
      console.error('Cannot delete edge: source or target node not found');
      return;
    }
    
    // Find the corresponding fact
    const factToDelete = facts.find(f => 
        f.subject === sourceNode.label && 
        f.object === targetNode.label && 
        f.predicate === edge.label
    );
    
    if (factToDelete) {
      try {
        // Delete the fact from backend - this will also remove it from the graph
        console.log('Deleting connection as fact from backend:', factToDelete);
        await deleteFact(factToDelete.id);
        console.log('Connection deleted as fact from backend');
        // deleteFact already calls refreshFacts, so we're done
      } catch (error) {
        console.error('Failed to delete connection as fact:', error);
        throw error; // Re-throw to let caller handle the error
      }
    } else {
      // If no fact found, just remove the edge from local state
      // But this shouldn't happen if graph and facts are in sync
      console.warn('No corresponding fact found for edge deletion - graph and facts may be out of sync');
    setEdges(prev => {
        const updated = prev.filter(e => e.id !== id);
      recalculateConnections(updated);
      return updated;
    });
      // Still refresh to ensure sync
      await refreshFacts();
    }
  }, [edges, nodes, facts, deleteFact, recalculateConnections, refreshFacts]);
    
  const addEdge = useCallback(async (sourceId: string, targetId: string, label: string = "related_to") => {
    const sourceNode = nodes.find(n => n.id === sourceId);
    const targetNode = nodes.find(n => n.id === targetId);
    
    if (!sourceNode || !targetNode) {
      console.error('Cannot create edge: source or target node not found');
      throw new Error('Source or target node not found');
    }
    
    // Create fact data for the connection
    const factData = {
        subject: sourceNode.label,
      predicate: label, // Use the provided label, not hardcoded "related_to"
        object: targetNode.label,
    };
    
    try {
      // Save to backend - this will check for duplicates and save to knowledge graph
      console.log('Adding connection as fact to backend:', factData);
      await addFact(factData);
      
      // Refresh facts to ensure graph is rebuilt with the new connection
      await refreshFacts();
      
      console.log('Connection saved as fact to backend');
    } catch (error) {
      console.error('Failed to save connection as fact:', error);
      // If it's a duplicate, the error will be handled by addFact
      throw error;
    }
  }, [nodes, addFact, refreshFacts]);

  const updateNodePosition = useCallback((nodeId: string, position: { x: number; y: number; z: number }) => {
    nodePositions.set(nodeId, position);
  }, [nodePositions]);

  const value: KnowledgeStoreContextType = {
    facts,
    nodes,
    edges,
    addFact,
    updateFact,
    deleteFact,
    updateNode,
    updateEdge,
    deleteEdge,
    addEdge,
    updateNodePosition,
    refreshFacts,
  };

  return (
    <KnowledgeStoreContext.Provider value={value}>
      {children}
    </KnowledgeStoreContext.Provider>
  );
}
