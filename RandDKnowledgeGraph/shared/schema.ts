import { z } from "zod";

export const sourceDocumentSchema = z.object({
  document: z.string(),
  uploadedAt: z.string().nullable().optional(),
});

export const factSchema = z.object({
  id: z.string(),
  subject: z.string(),
  predicate: z.string(),
  object: z.string(),
  source: z.string().optional(),
  details: z.string().optional(),
  sourceDocument: z.string().optional(),  // Backward compatibility: first source
  uploadedAt: z.string().optional(),  // Backward compatibility: first timestamp
  sourceDocuments: z.array(sourceDocumentSchema).optional(),  // New: all sources
  isInferred: z.boolean().optional(),  // Backward compatibility: marks if fact is inferred
  type: z.enum(["original", "inferred"]).optional(),  // New: type of fact ("original" or "inferred")
  confidence: z.number().min(0).max(1).optional(),  // New: confidence score (0.0 to 1.0)
  agent: z.string().optional(),  // Agent that processed the file (e.g., "PDF Agent", "CSV Agent")
});

export const insertFactSchema = factSchema.omit({ id: true });

export type Fact = z.infer<typeof factSchema>;
export type InsertFact = z.infer<typeof insertFactSchema>;

export const knowledgeBaseSchema = z.object({
  facts: z.array(factSchema),
  metadata: z.object({
    created: z.string(),
    updated: z.string(),
    version: z.string(),
    totalFacts: z.number(),
  }),
});

export type KnowledgeBase = z.infer<typeof knowledgeBaseSchema>;

export const documentSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.enum(["pdf", "docx", "txt", "csv", "pptx"]),
  size: z.number(),
  uploadedAt: z.string(),
  status: z.enum(["pending", "processing", "completed", "error"]),
});

export type Document = z.infer<typeof documentSchema>;

export const chatMessageSchema = z.object({
  id: z.string(),
  role: z.enum(["user", "assistant"]),
  content: z.string(),
  timestamp: z.string(),
});

export type ChatMessage = z.infer<typeof chatMessageSchema>;

export const graphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string(),
  connections: z.number(),
});

export const graphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  label: z.string(),
  details: z.string().optional(),
  sourceDocument: z.string().optional(),
  uploadedAt: z.string().optional(),
  isInferred: z.boolean().optional(),  // New: marks if edge represents an inferred fact
  agent: z.string().optional(),  // Agent that processed the file
});

export type GraphNode = z.infer<typeof graphNodeSchema>;
export type GraphEdge = z.infer<typeof graphEdgeSchema>;

// Simulator Schemas
export const experimentConfigSchema = z.object({
  name: z.string(),
  scenarioDescription: z.string(), // Natural language scenario description
  inputData: z.string().optional(), // Optional extra data not in knowledge base
  scenarioType: z.enum(["hypothesis", "prediction", "what_if", "validation"]),
  parameters: z.record(z.any()).optional(),
});

export const experimentResultSchema = z.object({
  id: z.string(),
  config: experimentConfigSchema,
  status: z.enum(["running", "completed", "failed"]),
  results: z.any().optional(),
  error: z.string().optional(),
  timestamp: z.string(),
});

export type ExperimentConfig = z.infer<typeof experimentConfigSchema>;
export type ExperimentResult = z.infer<typeof experimentResultSchema>;
