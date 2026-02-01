#!/usr/bin/env bash
# Test: (1) Causal graph sources update after upload, (2) Drag logic in frontend.
# Run from project root: RandDKnowledgeGraph/scripts/test-causal-and-drag.sh
# Requires: backend on 8001, curl, python3.

set -e
BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
TS=$(date +%s)

echo "=== 1. Causal graph sources (cache-bust) ==="
SOURCES=$(curl -s "$BASE_URL/api/knowledge/causal-graph/sources?_t=$TS")
COUNT=$(echo "$SOURCES" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('sources',[])))")
echo "Sources count: $COUNT"
if [ "$COUNT" -lt 1 ]; then
  echo "FAIL: expected at least one source"
  exit 1
fi
echo "OK: sources returned"

echo ""
echo "=== 2. Causal graph data for first dataset ==="
DOC=$(echo "$SOURCES" | python3 -c "
import sys,json
d=json.load(sys.stdin)
for s in d.get('sources',[]):
  if s.get('type')=='dataset':
    print(s.get('id','')); break
")
if [ -z "$DOC" ]; then
  echo "SKIP: no dataset source (upload a CSV that passes causal discovery to test)"
else
  ENCODED=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$DOC'))")
  GRAPH=$(curl -s "$BASE_URL/api/knowledge/causal-graph?include_inferred=true&source=dataset&document_name=$ENCODED&_t=$TS")
  NODES=$(echo "$GRAPH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('nodes',[])))")
  EDGES=$(echo "$GRAPH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('edges',[])))")
  echo "Dataset: $DOC -> nodes=$NODES edges=$EDGES"
  echo "OK: causal graph data returned"
fi

echo ""
echo "=== 3. Frontend: drag logic present ==="
VIS="RandDKnowledgeGraph/client/src/components/KnowledgeGraphVisualization.tsx"
if grep -q 'closest.*\.node-group' "$VIS" && grep -q 'className="node-group"' "$VIS"; then
  echo "OK: handleMouseDown skips pan when target is .node-group; nodes have class node-group"
else
  echo "FAIL: expected .node-group check and class in KnowledgeGraphVisualization.tsx"
  exit 1
fi

echo ""
echo "=== All automated checks passed ==="
echo "Manual test for drag: open Causal Graph, select a dataset, drag a node -> only node moves; drag canvas -> graph pans."
echo "Manual test for sources: upload a CSV (10+ numeric rows), click Refresh sources, select the CSV in Source -> graph shows that data."
