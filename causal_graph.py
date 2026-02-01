"""
Causal Graph Module
==================

Converts knowledge graph to causal graph by identifying and extracting
causal relationships (cause-and-effect) from the knowledge base.

Also supports data-driven causal discovery from CSV/tabular data using
the causal-learn library (PC algorithm).

A causal graph shows directed relationships where one entity causes,
leads to, or affects another entity.
"""

import re
import os
from typing import List, Dict, Tuple, Set, Optional, Any
from knowledge import graph, get_fact_details, get_fact_source_document
import rdflib
from urllib.parse import unquote


# Causal predicate patterns - these indicate causal relationships
CAUSAL_PREDICATES = {
    # Direct causal verbs
    'causes', 'cause', 'caused', 'causing',
    'leads to', 'lead to', 'led to', 'leading to',
    'results in', 'result in', 'resulted in', 'resulting in',
    'triggers', 'trigger', 'triggered', 'triggering',
    'induces', 'induce', 'induced', 'inducing',
    'produces', 'produce', 'produced', 'producing',
    'generates', 'generate', 'generated', 'generating',
    'creates', 'create', 'created', 'creating',
    
    # Effect/outcome verbs
    'affects', 'affect', 'affected', 'affecting',
    'influences', 'influence', 'influenced', 'influencing',
    'impacts', 'impact', 'impacted', 'impacting',
    'changes', 'change', 'changed', 'changing',
    'alters', 'alter', 'altered', 'altering',
    'modifies', 'modify', 'modified', 'modifying',
    
    # Increase/decrease patterns
    'increases', 'increase', 'increased', 'increasing',
    'decreases', 'decrease', 'decreased', 'decreasing',
    'enhances', 'enhance', 'enhanced', 'enhancing',
    'reduces', 'reduce', 'reduced', 'reducing',
    'improves', 'improve', 'improved', 'improving',
    'worsens', 'worsen', 'worsened', 'worsening',
    
    # Temporal/sequential patterns
    'follows', 'follow', 'followed', 'following',
    'precedes', 'precede', 'preceded', 'preceding',
    'enables', 'enable', 'enabled', 'enabling',
    'prevents', 'prevent', 'prevented', 'preventing',
    'blocks', 'block', 'blocked', 'blocking',
    'inhibits', 'inhibit', 'inhibited', 'inhibiting',
    
    # Conditional patterns
    'requires', 'require', 'required', 'requiring',
    'depends on', 'depend on', 'depended on', 'depending on',
    'determines', 'determine', 'determined', 'determining',
}

# Predicates that might be causal in context (so KB shows relationships when no strong causal verbs exist)
POTENTIALLY_CAUSAL = {
    'affects', 'influences', 'relates to', 'associated with',
    'correlated with', 'linked to', 'connected to',
    'part of', 'contains', 'includes', 'leads to', 'contributes to',
    'supports', 'enables', 'related to',
}

# Predicates that are NOT causal (exclude these)
NON_CAUSAL = {
    'is', 'are', 'was', 'were', 'has', 'have', 'had',
    'located in', 'based in', 'from', 'in', 'at',
    'collaborates with', 'works with', 'partners with',
    'similar to', 'same as', 'equals', 'equivalent to'
}


def is_causal_predicate(predicate: str) -> bool:
    """
    Check if a predicate indicates a causal relationship.
    
    Args:
        predicate: The predicate string to check
        
    Returns:
        True if the predicate indicates causality, False otherwise
    """
    predicate_lower = predicate.lower().strip()
    
    # Check exact matches
    if predicate_lower in CAUSAL_PREDICATES:
        return True
    
    # Check if predicate contains causal keywords
    for causal_keyword in CAUSAL_PREDICATES:
        if causal_keyword in predicate_lower:
            return True
    
    # Check for causal patterns
    causal_patterns = [
        r'\bcauses?\b',
        r'\bleads?\s+to\b',
        r'\bresults?\s+in\b',
        r'\btriggers?\b',
        r'\binduces?\b',
        r'\bproduces?\b',
        r'\bgenerates?\b',
        r'\bcreates?\b',
        r'\baffects?\b',
        r'\binfluences?\b',
        r'\bimpacts?\b',
        r'\bincreases?\b',
        r'\bdecreases?\b',
        r'\benhances?\b',
        r'\breduces?\b',
        r'\bimproves?\b',
        r'\bworsens?\b',
        r'\bprevents?\b',
        r'\bblocks?\b',
        r'\binhibits?\b',
        r'\benables?\b',
        r'\brequires?\b',
        r'\bdepends?\s+on\b',
        r'\bdetermines?\b',
    ]
    
    for pattern in causal_patterns:
        if re.search(pattern, predicate_lower):
            return True

    # Also treat potentially causal / relational predicates as weak causal (so KB shows something)
    if predicate_lower in POTENTIALLY_CAUSAL:
        return True
    for kw in POTENTIALLY_CAUSAL:
        if kw in predicate_lower:
            return True

    return False


def extract_causal_relationships() -> List[Dict]:
    """
    Extract causal relationships from the knowledge graph.
    
    Returns:
        List of causal relationship dictionaries with:
        - source: The cause entity
        - target: The effect entity
        - relationship: The causal predicate
        - details: Additional context/details
        - confidence: Confidence score (0.0 to 1.0)
        - source_document: Source document name
    """
    causal_relationships = []
    seen_relationships = set()  # Track duplicates
    
    # Iterate through all facts in the knowledge graph
    for subject, predicate, obj in graph:
        # Extract clean strings from RDF URIs
        subject_str = str(subject).split(':')[-1] if ':' in str(subject) else str(subject)
        predicate_str = str(predicate).split(':')[-1] if ':' in str(predicate) else str(predicate)
        object_str = str(obj).split(':')[-1] if ':' in str(obj) else str(obj)

        # Decode URI encoding so fact lookups and display match
        try:
            subject_str = unquote(subject_str.replace('_', ' '))
            predicate_str = unquote(predicate_str.replace('_', ' '))
            object_str = unquote(object_str.replace('_', ' '))
        except Exception:
            pass
        
        # Check if this is a causal relationship
        if is_causal_predicate(predicate_str):
            # Skip non-causal predicates
            if predicate_str.lower() in NON_CAUSAL:
                continue
            
            # Create relationship key for deduplication
            rel_key = (subject_str.lower(), predicate_str.lower(), object_str.lower())
            if rel_key in seen_relationships:
                continue
            seen_relationships.add(rel_key)
            
            # Get additional details
            details = get_fact_details(subject_str, predicate_str, object_str)
            source_docs = get_fact_source_document(subject_str, predicate_str, object_str)
            # source_docs returns list of (source_document, uploaded_at) tuples
            source_doc = source_docs[0][0] if source_docs and len(source_docs) > 0 else "unknown"
            
            # Determine confidence based on predicate strength
            confidence = 0.7  # Default
            predicate_lower = predicate_str.lower()
            if any(strong in predicate_lower for strong in ['causes', 'triggers', 'induces', 'produces']):
                confidence = 0.9
            elif any(medium in predicate_lower for medium in ['leads to', 'results in', 'creates', 'generates']):
                confidence = 0.8
            elif any(weak in predicate_lower for weak in ['affects', 'influences', 'impacts', 'changes']):
                confidence = 0.6
            elif any(kw in predicate_lower for kw in POTENTIALLY_CAUSAL):
                confidence = 0.5  # Weak causal / relational

            causal_relationships.append({
                'source': subject_str,
                'target': object_str,
                'relationship': predicate_str,
                'details': details or f"{subject_str} {predicate_str} {object_str}",
                'confidence': confidence,
                'source_document': source_doc
            })
    
    return causal_relationships


def infer_causal_relationships(causal_rels: List[Dict]) -> List[Dict]:
    """
    Infer additional causal relationships through transitive reasoning.
    
    Example: If A causes B and B causes C, then A indirectly causes C.
    
    Args:
        causal_rels: List of direct causal relationships
        
    Returns:
        List of inferred causal relationships
    """
    inferred = []
    
    # Build a directed graph of causal relationships
    causal_graph = {}
    for rel in causal_rels:
        source = rel['source']
        target = rel['target']
        if source not in causal_graph:
            causal_graph[source] = []
        causal_graph[source].append(target)
    
    # Find transitive causal paths (A -> B -> C implies A -> C)
    for source in causal_graph:
        visited = set()
        queue = [(source, [source])]  # (current_node, path)
        
        while queue:
            current, path = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # If path length > 2, we have a transitive relationship
            if len(path) > 2:
                # Create inferred relationship from first to last
                inferred_source = path[0]
                inferred_target = path[-1]
                
                # Check if this relationship already exists
                exists = any(
                    r['source'] == inferred_source and r['target'] == inferred_target
                    for r in causal_rels + inferred
                )
                
                if not exists:
                    # Calculate confidence based on path length (longer paths = lower confidence)
                    path_confidence = max(0.3, 0.7 - (len(path) - 2) * 0.1)
                    
                    inferred.append({
                        'source': inferred_source,
                        'target': inferred_target,
                        'relationship': 'indirectly causes',
                        'details': f"Inferred from path: {' -> '.join(path)}",
                        'confidence': path_confidence,
                        'source_document': 'inferred',
                        'is_inferred': True
                    })
            
            # Continue exploring
            if current in causal_graph:
                for neighbor in causal_graph[current]:
                    if neighbor not in path:  # Avoid cycles
                        queue.append((neighbor, path + [neighbor]))
    
    return inferred


# ---------------------------------------------------------------------------
# Data-driven causal discovery (causal-learn)
# ---------------------------------------------------------------------------

# Edge type constants in causal-learn graph matrix
# 0 = no edge, 1 = tail (arrow from), -1 = head (arrow to), 2 = circle (undirected)
_EDGE_TAIL = 1
_EDGE_HEAD = -1
_EDGE_NONE = 0


def discover_causal_structure_from_csv(
    csv_path: str,
    delimiter: str = ";",
    alpha: float = 0.05,
    max_num_vars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run causal discovery on a CSV file using causal-learn (PC algorithm).
    
    Uses only numeric columns. Returns nodes (variable names) and directed
    edges in the same format as get_causal_graph_data for UI visualization.
    
    Args:
        csv_path: Path to the CSV file
        delimiter: CSV delimiter (default ';' for European-style CSVs)
        alpha: Significance level for independence tests (default 0.05)
        max_num_vars: If set, use only first N numeric columns (for large CSVs)
        
    Returns:
        Dict with keys: nodes, edges, stats, variable_names, error (if any)
    """
    import pandas as pd
    import numpy as np

    result = {
        "nodes": [],
        "edges": [],
        "stats": {
            "total_nodes": 0,
            "total_edges": 0,
            "direct_relationships": 0,
            "inferred_relationships": 0,
            "most_causal_entities": [],
            "algorithm": "PC",
            "source": "data",
        },
        "variable_names": [],
    }

    try:
        # Try multiple delimiters and decimals so different CSV formats work (comma vs semicolon, . vs ,)
        df = None
        for sep, dec in [(",", "."), (";", ","), (";", "."), (",", ","), ("\t", ".")]:
            try:
                df = pd.read_csv(
                    csv_path,
                    delimiter=sep,
                    encoding="utf-8",
                    on_bad_lines="skip",
                    decimal=dec,
                )
                if df is not None and len(df.columns) >= 2:
                    break
            except Exception:
                continue
        if df is None or len(df.columns) < 2:
            result["error"] = "Could not read CSV with multiple columns (try comma or semicolon delimiter)"
            return result
        # Flatten column names (e.g. "Fine\nAggregate" -> "Fine Aggregate")
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        # Drop completely non-numeric columns
        numeric_cols = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 2:  # at least 2 valid numbers
                numeric_cols.append(c)
        if not numeric_cols:
            result["error"] = "No numeric columns found in CSV"
            return result
        df = df[numeric_cols].copy()
        df = df.dropna(how="all", axis=1)
        df = df.dropna(how="all", axis=0)
        if max_num_vars and len(df.columns) > max_num_vars:
            df = df.iloc[:, :max_num_vars]
        numeric_cols = list(df.columns)
        # Ensure numeric dtypes
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if len(df) < 10:
            result["error"] = "Too few rows after dropping NaN (need at least 10)"
            return result

        data = df.values.astype(np.float64)
        n_vars = data.shape[1]
        var_names = numeric_cols

        # Run PC algorithm (causal-learn)
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz

        cg = pc(data, alpha=alpha, indep_test=fisherz, show_progress=False)

        # cg.G is the graph; cg.G.graph is the adjacency matrix
        # In causal-learn: (i,j)=1 tail at i, (i,j)=-1 head at j => edge i -> j
        # So (i,j)==1 and (j,i)==-1 means directed edge from i to j
        g = cg.G.graph
        if g is None:
            result["error"] = "PC returned no graph"
            result["nodes"] = [{"id": f"node_{i}", "label": var_names[i], "type": "variable", "connections": 0} for i in range(n_vars)]
            result["stats"]["total_nodes"] = n_vars
            return result

        nodes_set = set(range(n_vars))
        edges_list = []  # (from_idx, to_idx)
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                # causal-learn: (i,j)=-1 (head at j) and (j,i)=1 (tail at i) => directed edge i -> j
                val_ij = int(g[i, j]) if hasattr(g, "__getitem__") else 0
                val_ji = int(g[j, i]) if hasattr(g, "__getitem__") else 0
                if val_ij == _EDGE_HEAD and val_ji == _EDGE_TAIL:
                    edges_list.append((i, j))  # i -> j
                elif val_ij == _EDGE_TAIL and val_ji == _EDGE_HEAD:
                    edges_list.append((j, i))  # j -> i
                # Undirected/circle: show as directed for display (pick one direction)
                elif val_ij != _EDGE_NONE or val_ji != _EDGE_NONE:
                    if (j, i) not in edges_list and (i, j) not in edges_list:
                        edges_list.append((i, j))

        # Ensure acyclic: keep only edges that do not create cycles
        edges_list = _make_acyclic(edges_list, n_vars)

        node_connections = {i: 0 for i in range(n_vars)}
        for i, j in edges_list:
            node_connections[i] = node_connections.get(i, 0) + 1
            node_connections[j] = node_connections.get(j, 0) + 1

        nodes = []
        for i in range(n_vars):
            name = var_names[i]
            node_id = f"node_{_sanitize_id(name)}"
            nodes.append({
                "id": node_id,
                "label": name,
                "type": "variable",
                "connections": node_connections.get(i, 0),
            })
        edges = []
        for idx, (i, j) in enumerate(edges_list):
            src_name = var_names[i]
            tgt_name = var_names[j]
            edges.append({
                "id": f"edge_{idx}",
                "source": f"node_{_sanitize_id(src_name)}",
                "target": f"node_{_sanitize_id(tgt_name)}",
                "label": "causes",
                "details": f"Data-driven: {src_name} → {tgt_name}",
                "confidence": 0.8,
                "is_inferred": False,
                "source_document": "data",
            })

        result["nodes"] = nodes
        result["edges"] = edges
        result["variable_names"] = var_names
        result["stats"]["total_nodes"] = len(nodes)
        result["stats"]["total_edges"] = len(edges)
        result["stats"]["direct_relationships"] = len(edges)
        result["stats"]["inferred_relationships"] = 0
        most_causal = sorted(
            [(var_names[i], node_connections.get(i, 0)) for i in range(n_vars)],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        result["stats"]["most_causal_entities"] = [{"entity": name, "connections": count} for name, count in most_causal]
        # Persist numeric data for DoWhy causal inference (limit rows to keep store size reasonable)
        _MAX_DOWHY_ROWS = 5000
        result["data_columns"] = var_names
        result["data_rows"] = df.head(_MAX_DOWHY_ROWS).values.tolist()
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def _make_acyclic(edges_list: List[Tuple[int, int]], n_vars: int) -> List[Tuple[int, int]]:
    """Return a subset of edges that forms a DAG (no cycles)."""
    # Build adjacency list for the current graph; only add edge (u,v) if no path v -> u
    adj: Dict[int, List[int]] = {i: [] for i in range(n_vars)}

    def has_path(from_node: int, to_node: int, visited: Set[int]) -> bool:
        if from_node == to_node:
            return True
        visited.add(from_node)
        for w in adj.get(from_node, []):
            if w not in visited and has_path(w, to_node, visited):
                return True
        return False

    acyclic = []
    for (u, v) in edges_list:
        if has_path(v, u, set()):
            continue  # would create cycle
        acyclic.append((u, v))
        adj[u].append(v)
    return acyclic


def _sanitize_id(name: str) -> str:
    """Turn variable name into a safe node id (no spaces, no special chars)."""
    return re.sub(r"[^\w\-]", "_", str(name).strip()).strip("_") or "var"


# Storage for data-driven causal graphs (keyed by document name)
# Use module directory only so path does not depend on CWD (same as knowledge/documents_store)
_STORE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_FILENAME = "causal_graphs_store.json"
CAUSAL_STORE_PATH = os.path.join(_STORE_DIR, STORE_FILENAME)


def _get_store_path() -> str:
    """Return path to causal graphs store (always module dir)."""
    return CAUSAL_STORE_PATH


def _load_causal_graphs_store() -> Dict[str, Any]:
    path = _get_store_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                import json
                return json.load(f)
        except Exception:
            pass
    return {"graphs": {}, "last_updated": None}


def _save_causal_graphs_store(store: Dict[str, Any]) -> None:
    import json
    from datetime import datetime
    store["last_updated"] = datetime.now().isoformat()
    path = _get_store_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def save_data_driven_causal_graph(document_name: str, graph_data: Dict[str, Any]) -> None:
    """Store data-driven causal graph (and data for DoWhy) for a document (e.g. CSV filename)."""
    store = _load_causal_graphs_store()
    entry = {
        "nodes": graph_data.get("nodes", []),
        "edges": graph_data.get("edges", []),
        "stats": graph_data.get("stats", {}),
    }
    if graph_data.get("data_columns") and graph_data.get("data_rows") is not None:
        entry["data_columns"] = graph_data["data_columns"]
        entry["data_rows"] = graph_data["data_rows"]
    store["graphs"][document_name] = entry
    _save_causal_graphs_store(store)


def get_data_driven_causal_graph(document_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve data-driven causal graph for a document."""
    store = _load_causal_graphs_store()
    return store.get("graphs", {}).get(document_name)


def run_dowhy_effect_estimation(
    document_name: str,
    treatment: str,
    outcome: str,
    method: str = "backdoor.linear_regression",
) -> Dict[str, Any]:
    """
    Run DoWhy causal effect estimation on a stored data-driven causal graph.
    Requires graph + data to be stored (from CSV upload). Returns estimate and refutation.
    """
    import pandas as pd
    import numpy as np

    out = {
        "success": False,
        "treatment": treatment,
        "outcome": outcome,
        "estimate": None,
        "estimate_value": None,
        "refutation": None,
        "error": None,
    }
    try:
        graph_data = get_data_driven_causal_graph(document_name)
        if not graph_data:
            out["error"] = f"No data-driven graph found for document: {document_name}"
            return out
        cols = graph_data.get("data_columns")
        rows = graph_data.get("data_rows")
        if not cols or rows is None:
            out["error"] = "No data stored for this dataset (re-upload the CSV to enable DoWhy)"
            return out
        df = pd.DataFrame(rows, columns=cols)
        if treatment not in df.columns:
            out["error"] = f"Treatment '{treatment}' is not a column in the data"
            return out
        if outcome not in df.columns:
            out["error"] = f"Outcome '{outcome}' is not a column in the data"
            return out

        import networkx as nx
        node_id_to_label = {n["id"]: n["label"] for n in graph_data.get("nodes", [])}
        nx_graph = nx.DiGraph()
        for e in graph_data.get("edges", []):
            src = node_id_to_label.get(e["source"], e["source"].replace("node_", "").replace("_", " "))
            tgt = node_id_to_label.get(e["target"], e["target"].replace("node_", "").replace("_", " "))
            if src in df.columns and tgt in df.columns:
                nx_graph.add_edge(src, tgt)
        for c in df.columns:
            if c not in nx_graph:
                nx_graph.add_node(c)

        from dowhy import CausalModel
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=nx_graph,
        )
        identified = model.identify_effect()
        estimate = model.estimate_effect(identified, method_name=method)
        out["estimate_value"] = float(estimate.value) if estimate.value is not None else None
        out["estimate"] = str(estimate)
        try:
            refute = model.refute_estimate(identified, estimate, method_name="random_common_cause")
            out["refutation"] = str(refute)
        except Exception:
            out["refutation"] = "(refutation skipped)"
        out["success"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def list_data_driven_causal_graph_sources() -> List[str]:
    """List document names that have a stored data-driven causal graph."""
    store = _load_causal_graphs_store()
    names = list(store.get("graphs", {}).keys())
    if names:
        print(f"📊 Causal graph store: {_get_store_path()} → {len(names)} dataset(s): {names[:5]}{'...' if len(names) > 5 else ''}")
    return names


def clear_data_driven_causal_graphs() -> None:
    """Clear all data-driven causal graphs (call on backend startup).
    After restart, only CSVs uploaded in the current session will appear in the causal graph source list."""
    store = {"graphs": {}, "last_updated": None}
    _save_causal_graphs_store(store)
    path = _get_store_path()
    print(f"🧹 Causal graph store cleared at {path} — only datasets uploaded this session will appear")


def get_causal_graph_data(include_inferred: bool = True) -> Dict:
    """
    Get causal graph data in format suitable for visualization.
    
    Args:
        include_inferred: Whether to include inferred causal relationships
        
    Returns:
        Dictionary with:
        - nodes: List of unique entities in causal relationships
        - edges: List of causal relationships
        - stats: Statistics about the causal graph
    """
    # Extract direct causal relationships
    causal_rels = extract_causal_relationships()
    
    # Optionally infer additional relationships
    inferred_rels = []
    if include_inferred:
        inferred_rels = infer_causal_relationships(causal_rels)
    
    all_rels = causal_rels + inferred_rels
    
    # Extract unique nodes
    nodes_set = set()
    for rel in all_rels:
        nodes_set.add(rel['source'])
        nodes_set.add(rel['target'])
    
    # Create node list with metadata
    nodes = []
    node_connections = {}
    for rel in all_rels:
        source = rel['source']
        target = rel['target']
        node_connections[source] = node_connections.get(source, 0) + 1
        node_connections[target] = node_connections.get(target, 0) + 1
    
    for node_name in nodes_set:
        nodes.append({
            'id': f"node_{node_name.lower().replace(' ', '_')}",
            'label': node_name,
            'type': 'entity',
            'connections': node_connections.get(node_name, 0)
        })
    
    # Create edge list
    edges = []
    for i, rel in enumerate(all_rels):
        source_id = f"node_{rel['source'].lower().replace(' ', '_')}"
        target_id = f"node_{rel['target'].lower().replace(' ', '_')}"
        
        edges.append({
            'id': f"edge_{i}",
            'source': source_id,
            'target': target_id,
            'label': rel['relationship'],
            'details': rel.get('details'),
            'confidence': rel.get('confidence', 0.7),
            'is_inferred': rel.get('is_inferred', False),
            'source_document': rel.get('source_document', 'unknown')
        })
    
    # Calculate statistics
    direct_count = len(causal_rels)
    inferred_count = len(inferred_rels)
    total_nodes = len(nodes)
    total_edges = len(edges)
    
    # Find most connected nodes (causes/effects)
    most_causal = sorted(
        [(node, node_connections.get(node, 0)) for node in nodes_set],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return {
        'nodes': nodes,
        'edges': edges,
        'stats': {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'direct_relationships': direct_count,
            'inferred_relationships': inferred_count,
            'most_causal_entities': [{'entity': name, 'connections': count} for name, count in most_causal]
        }
    }
