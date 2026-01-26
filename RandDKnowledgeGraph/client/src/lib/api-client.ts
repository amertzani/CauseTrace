/**
 * API Client for Local FastAPI Backend Integration
 *
 * This file contains all the API calls to your local FastAPI backend.
 * The backend runs on port 8001 by default (configurable via API_PORT env var).
 */

// Local backend URL - your FastAPI server
// Defaults to port 8001 (backend uses this port by default)
// Override with VITE_API_URL environment variable if needed
const BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8001";

// Use local backend directly (not HuggingFace or proxy)
const USE_LOCAL_BACKEND = true;
const USE_LOCAL_STORAGE = false; // Fallback to local storage if backend fails
const USE_BACKEND_PROXY = false; // Don't use Replit proxy - connect directly

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Local storage helper for fallback
class LocalKnowledgeStore {
  private storageKey = "research_brain_knowledge";

  getFacts(): any[] {
    const data = localStorage.getItem(this.storageKey);
    return data ? JSON.parse(data) : [];
  }

  saveFacts(facts: any[]): void {
    localStorage.setItem(this.storageKey, JSON.stringify(facts));
  }

  addFact(fact: any): any {
    const facts = this.getFacts();
    const newFact = {
      ...fact,
      id: String(facts.length + 1),
    };
    facts.push(newFact);
    this.saveFacts(facts);
    return newFact;
  }

  updateFact(id: string, updates: any): any | null {
    const facts = this.getFacts();
    const index = facts.findIndex((f) => String(f.id) === String(id));
    if (index !== -1) {
      facts[index] = { ...facts[index], ...updates };
      this.saveFacts(facts);
      return facts[index];
    }
    return null;
  }

  deleteFact(id: string): any | null {
    const facts = this.getFacts();
    const index = facts.findIndex((f) => String(f.id) === String(id));
    if (index !== -1) {
      const deleted = facts.splice(index, 1)[0];
      this.saveFacts(facts);
      return deleted;
    }
    return null;
  }
}

const localStore = new LocalKnowledgeStore();

class HuggingFaceApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  /**
   * Call a Gradio API function
   */
  private async callGradioApi<T>(
    functionName: string,
    params: any[] = [],
  ): Promise<ApiResponse<T>> {
    try {
      // Call the function
      const callResponse = await fetch(`${this.baseUrl}/call/${functionName}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: params }),
      });

      if (!callResponse.ok) {
        throw new Error(`HTTP error! status: ${callResponse.status}`);
      }

      const callData = await callResponse.json();
      const eventId = callData.event_id;

      // Poll for the result
      const statusResponse = await fetch(
        `${this.baseUrl}/call/${functionName}/${eventId}`,
      );

      if (!statusResponse.ok) {
        throw new Error(`HTTP error! status: ${statusResponse.status}`);
      }

      // Read the streaming response
      const reader = statusResponse.body?.getReader();
      const decoder = new TextDecoder();
      let result: any = null;

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.msg === "process_completed") {
                  result = data.output?.data?.[0];
                }
              } catch (e) {
                // Ignore parse errors
              }
            }
          }
        }
      }

      return { success: true, data: result };
    } catch (error) {
      console.error("Gradio API call failed:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    timeoutMs: number = 90000, // Default 90 seconds
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      console.log(`📡 API Request: ${options.method || 'GET'} ${url} (timeout: ${timeoutMs}ms)`);
      
      // Don't override Content-Type if it's FormData (for file uploads)
      const headers: Record<string, string> = {
        ...(options.headers as Record<string, string>),
      };
      
      if (!(options.body instanceof FormData)) {
        headers["Content-Type"] = "application/json";
      }

      // Create timeout controller (AbortSignal.timeout might not be available in all browsers)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      
      const fetchOptions: RequestInit = {
        ...options,
        headers,
        signal: controller.signal,
      };

      const response = await fetch(url, fetchOptions);
      clearTimeout(timeoutId); // Clear timeout if request succeeds

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ API Error: ${response.status} - ${errorText}`);
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      console.log(`✅ API Success: ${url}`, data);
      // FastAPI returns data directly, wrap it if needed
      return { success: true, data };
    } catch (error) {
      console.error(`❌ API request failed: ${endpoint}`, error);
      if (error instanceof Error) {
        if (error.name === 'AbortError' || error.message.includes('timeout') || error.message.includes('aborted')) {
          return {
            success: false,
            error: "Request timed out. Is the backend running at " + this.baseUrl + "?",
          };
        }
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
          return {
            success: false,
            error: "Cannot connect to backend at " + this.baseUrl + ". Check if backend is running.",
          };
        }
      }
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  // Document Upload - Updated for FastAPI
  async uploadDocuments(files: File[]): Promise<ApiResponse<any>> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    // Calculate timeout based on file size (5 minutes base + 1 minute per MB)
    const totalSizeMB = files.reduce((sum, file) => sum + file.size, 0) / (1024 * 1024);
    const timeoutMs = Math.max(300000, 300000 + totalSizeMB * 60000); // Min 5 min, +1 min per MB
    console.log(`📤 Uploading ${files.length} file(s), total ${totalSizeMB.toFixed(2)}MB, timeout: ${(timeoutMs/1000).toFixed(0)}s`);

    return this.request("/api/knowledge/upload", {
      method: "POST",
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    }, timeoutMs);
  }

  // Process Documents
  async processDocuments(documentIds: string[]): Promise<ApiResponse<any>> {
    // Processing can take a long time for large documents - use 10 minute timeout
    return this.request("/api/process", {
      method: "POST",
      body: JSON.stringify({ document_ids: documentIds }),
    }, 600000); // 10 minutes for document processing
  }

  // Knowledge Base - Updated for FastAPI
  async getKnowledgeBase(
    includeInferred: boolean = true,
    minConfidence: number = 0.0
  ): Promise<ApiResponse<any>> {
    console.log('📡 API: getKnowledgeBase called', { includeInferred, minConfidence });
    if (USE_LOCAL_STORAGE) {
      // Use local storage fallback
      console.log('📡 API: Using local storage');
      return { success: true, data: { facts: localStore.getFacts() } };
    }
    
    if (USE_LOCAL_BACKEND) {
      // Use local FastAPI backend - get structured facts
      console.log('📡 API: Using local backend, calling GET /api/knowledge/facts');
      const response = await this.request("/api/knowledge/facts", { method: "GET" });
      console.log('📡 API: GET response:', JSON.stringify(response, null, 2));
      
      // FastAPI returns data directly, check if facts are in data.facts or just data
      if (response.success) {
        // FastAPI returns { facts: [...], total_facts: X, status: "success" }
        // The request() method wraps it in { success: true, data: { facts: [...], ... } }
        const facts = response.data?.facts || (Array.isArray(response.data) ? response.data : []);
        console.log(`📡 API: Extracted ${Array.isArray(facts) ? facts.length : 'non-array'} facts from response`);
        console.log('📡 API: Response data keys:', response.data ? Object.keys(response.data) : 'no data');
        
        if (Array.isArray(facts) && facts.length > 0) {
          console.log(`📡 API: Successfully retrieved ${facts.length} facts`);
          console.log('📡 API: First fact:', facts[0]);
          return { success: true, data: { facts } };
        } else {
          console.log('📡 API: No facts in response. Response data:', JSON.stringify(response.data, null, 2));
          return { success: true, data: { facts: [] } };
        }
      }
      console.log('📡 API: Response not successful:', response);
      return response;
    }
    
    if (USE_BACKEND_PROXY) {
      // Use backend proxy (fallback)
      console.log('📡 API: Using backend proxy');
      return this.request("/api/hf/knowledge-base", { method: "GET" });
    }
    
    console.log('📡 API: Using Gradio API fallback');
    return this.callGradioApi("api_get_knowledge_base", []);
  }

  async createFact(fact: any): Promise<ApiResponse<any>> {
    console.log('📡 API: createFact called with:', fact);
    if (USE_LOCAL_STORAGE) {
      // Use local storage fallback
      const newFact = localStore.addFact(fact);
      return { success: true, data: { success: true, fact: newFact } };
    }
    
    if (USE_LOCAL_BACKEND) {
      // Use local FastAPI backend - use structured fact endpoint
      if (!fact.subject || !fact.predicate || fact.object === undefined) {
        return { success: false, error: "Fact must have subject, predicate, and object" };
      }
      
      console.log('📡 API: POST /api/knowledge/facts with data:', {
        subject: fact.subject,
        predicate: fact.predicate,
        object: fact.object,
        source: fact.source || "manual"
      });
      
      const response = await this.request("/api/knowledge/facts", {
        method: "POST",
        body: JSON.stringify({
          subject: fact.subject,
          predicate: fact.predicate,
          object: fact.object,
          source: fact.source || "manual"
        }),
      });
      
      console.log('📡 API: createFact response:', response);
      
      // Backend returns { fact, status, total_facts } - wrap it properly
      if (response.success && response.data?.fact) {
        return { success: true, data: { fact: response.data.fact } };
      }
      return response;
    }
    
    if (USE_BACKEND_PROXY) {
      // Use backend proxy (fallback)
      return this.request("/api/hf/facts", {
        method: "POST",
        body: JSON.stringify(fact),
      });
    }
    
    return this.callGradioApi("api_create_fact", [
      fact.subject || "",
      fact.predicate || "",
      fact.object || "",
      fact.source || "API",
    ]);
  }

  async updateFact(factId: string, updates: any): Promise<ApiResponse<any>> {
    if (USE_LOCAL_STORAGE) {
      // Use local storage fallback
      const updated = localStore.updateFact(factId, updates);
      if (updated) {
        return { success: true, data: { success: true, fact: updated } };
      }
      return { success: false, error: "Fact not found" };
    }
    
    // Note: FastAPI backend doesn't have update endpoint yet
    // For now, delete and recreate, or use local storage
    if (USE_LOCAL_BACKEND) {
      // FastAPI doesn't support update yet - return error
      return { success: false, error: "Update not yet supported by local backend. Use delete and create instead." };
    }
    
    if (USE_BACKEND_PROXY) {
      // Use backend proxy (fallback)
      return this.request(`/api/hf/facts/${factId}`, {
        method: "PATCH",
        body: JSON.stringify(updates),
      });
    }
    
    return this.callGradioApi("api_update_fact", [
      factId,
      updates.subject || "",
      updates.predicate || "",
      updates.object || "",
    ]);
  }

  async deleteFact(factId: string, factData?: { subject: string; predicate: string; object: string }): Promise<ApiResponse<any>> {
    if (USE_LOCAL_STORAGE) {
      // Use local storage fallback
      const deleted = localStore.deleteFact(factId);
      if (deleted) {
        return { success: true, data: { success: true, deleted } };
      }
      return { success: false, error: "Fact not found" };
    }
    
    if (USE_LOCAL_BACKEND) {
      // Use local FastAPI backend - delete by fact data (subject, predicate, object)
      // factId can be the fact ID or we use factData if provided
      if (factData) {
        // Create JSON string with fact data
        const factJson = JSON.stringify(factData);
        console.log('📡 API: deleteFact called with fact data:', factData);
        const response = await this.request(`/api/knowledge/facts/${encodeURIComponent(factJson)}`, {
          method: "DELETE",
        });
        console.log('📡 API: deleteFact response:', response);
        return response;
      } else {
        // Fallback: try to use factId as keyword
        return this.request("/api/knowledge/delete", {
          method: "DELETE",
          body: JSON.stringify({ keyword: factId }),
        });
      }
    }
    
    if (USE_BACKEND_PROXY) {
      // Use backend proxy (fallback)
      return this.request(`/api/hf/facts/${factId}`, {
        method: "DELETE",
      });
    }
    
    return this.callGradioApi("api_delete_fact", [factId]);
  }

  // Knowledge Graph - Updated for FastAPI
  async getKnowledgeGraph(): Promise<ApiResponse<any>> {
    if (USE_LOCAL_STORAGE) {
      // Build graph from local facts
      const facts = localStore.getFacts();
      const nodes: any[] = [];
      const edges: any[] = [];
      const nodeSet = new Set<string>();

      facts.forEach((fact: any) => {
        const subj = fact.subject;
        const obj = fact.object;
        const pred = fact.predicate;

        if (subj && !nodeSet.has(subj)) {
          nodes.push({ id: subj, label: subj, type: "concept" });
          nodeSet.add(subj);
        }

        if (obj && !nodeSet.has(obj)) {
          nodes.push({ id: obj, label: obj, type: "entity" });
          nodeSet.add(obj);
        }

        if (subj && obj && pred) {
          edges.push({
            id: `${subj}-${pred}-${obj}`,
            source: subj,
            target: obj,
            label: pred,
          });
        }
      });

      return { success: true, data: { nodes, edges } };
    }
    
    if (USE_LOCAL_BACKEND) {
      // Use local FastAPI backend
      const response = await this.request("/api/knowledge/graph", { method: "GET" });
      if (response.success && response.data?.graph_html) {
        // Return HTML for rendering
        return { success: true, data: { html: response.data.graph_html } };
      }
      return response;
    }
    
    if (USE_BACKEND_PROXY) {
      // Use backend proxy (fallback)
      return this.request("/api/hf/graph", { method: "GET" });
    }
    
    return this.callGradioApi("api_get_graph", []);
  }

  // Import/Export
  async exportKnowledgeBase(includeInferred: boolean = true, minConfidence: number = 0.0): Promise<ApiResponse<any>> {
    const params = new URLSearchParams({
      include_inferred: includeInferred.toString(),
      min_confidence: minConfidence.toString(),
    });
    return this.request(`/api/export?${params.toString()}`, {
      method: "GET",
    });
  }

  async importKnowledgeBase(data: any): Promise<ApiResponse<any>> {
    console.log('📡 API: importKnowledgeBase called with data:', data);
    if (USE_LOCAL_BACKEND) {
      // Create a FormData with the JSON file
      const formData = new FormData();
      const jsonBlob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      formData.append('file', jsonBlob, 'knowledge-base.json');
      
      const response = await this.request("/api/knowledge/import", {
        method: "POST",
        body: formData,
      });
      console.log('📡 API: importKnowledgeBase response:', response);
      return response;
    }
    // Fallback for other modes
    return this.request("/api/knowledge/import", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // Chat/LLM - Updated for FastAPI
  async sendChatMessage(
    message: string,
    history: any[] = [],
  ): Promise<ApiResponse<any>> {
    if (USE_LOCAL_BACKEND) {
      // Use local FastAPI backend
      const response = await this.request("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message, history }),
      });
      // FastAPI returns { response: "...", status: "success" }
      if (response.success && response.data?.response) {
        return { success: true, data: response.data.response };
      }
      return response;
    }
    
    // Fallback to old endpoint
    return this.request("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, history }),
    });
  }

  // Documents
  async getDocuments(): Promise<ApiResponse<any>> {
    console.log('📡 API: getDocuments called');
    if (USE_LOCAL_BACKEND) {
      const response = await this.request("/api/documents", { method: "GET" });
      console.log('📡 API: getDocuments response:', response);
      if (response.success && response.data?.documents) {
        return { success: true, data: { documents: response.data.documents } };
      }
      return response;
    }
    // Fallback for other modes
    return this.request("/api/documents", { method: "GET" });
  }

  async deleteDocument(documentId: string): Promise<ApiResponse<any>> {
    return this.request(`/api/documents/${documentId}`, {
      method: "DELETE",
    });
  }

  // Simulator - Run Experiment
  async runExperiment(experimentConfig: any): Promise<ApiResponse<any>> {
    console.log('📡 API: runExperiment called with:', experimentConfig);
    if (USE_LOCAL_BACKEND) {
      const response = await this.request("/api/simulator/experiment", {
        method: "POST",
        body: JSON.stringify(experimentConfig),
      }, 300000); // 5 minutes timeout for experiments
      console.log('📡 API: runExperiment response:', response);
      return response;
    }
    return {
      success: false,
      error: "Simulator requires local backend",
    };
  }

  // Simulator - Test Scenario
  async testScenario(scenarioData: any): Promise<ApiResponse<any>> {
    console.log('📡 API: testScenario called with:', scenarioData);
    if (USE_LOCAL_BACKEND) {
      const response = await this.request("/api/simulator/scenario", {
        method: "POST",
        body: JSON.stringify(scenarioData),
      }, 300000); // 5 minutes timeout for scenarios
      console.log('📡 API: testScenario response:', response);
      return response;
    }
    return {
      success: false,
      error: "Simulator requires local backend",
    };
  }

  // Simulator - Get Experiments
  async getExperiments(): Promise<ApiResponse<any>> {
    console.log('📡 API: getExperiments called');
    if (USE_LOCAL_BACKEND) {
      const response = await this.request("/api/simulator/experiments", {
        method: "GET",
      });
      console.log('📡 API: getExperiments response:', response);
      return response;
    }
    return {
      success: true,
      data: { experiments: [] },
    };
  }

  // Simulator - Delete Experiment
  async deleteExperiment(experimentId: string): Promise<ApiResponse<any>> {
    console.log('📡 API: deleteExperiment called with:', experimentId);
    if (USE_LOCAL_BACKEND) {
      const response = await this.request(`/api/simulator/experiments/${experimentId}`, {
        method: "DELETE",
      });
      console.log('📡 API: deleteExperiment response:', response);
      return response;
    }
    return {
      success: false,
      error: "Simulator requires local backend",
    };
  }
}

// Export singleton instance
export const hfApi = new HuggingFaceApiClient(BASE_URL);

// Export types for use in components
export type { ApiResponse };
