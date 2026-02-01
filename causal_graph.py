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
    'supports', 'enables', 'related to', 'belongs to',
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


def extract_all_relationships_as_weak_causal() -> List[Dict]:
    """
    When no strong causal predicates are found, treat all KB relationships as weak causal
    so the Causal Graph and Causal Relationships pages still show data.
    Skips only NON_CAUSAL predicates (e.g. 'is', 'has', 'located in').
    """
    relationships = []
    seen = set()
    for subject, predicate, obj in graph:
        subject_str = str(subject).split(':')[-1] if ':' in str(subject) else str(subject)
        predicate_str = str(predicate).split(':')[-1] if ':' in str(predicate) else str(predicate)
        object_str = str(obj).split(':')[-1] if ':' in str(obj) else str(obj)
        try:
            subject_str = unquote(subject_str.replace('_', ' '))
            predicate_str = unquote(predicate_str.replace('_', ' '))
            object_str = unquote(object_str.replace('_', ' '))
        except Exception:
            pass
        pred_lower = predicate_str.lower().strip()
        if pred_lower in NON_CAUSAL:
            continue
        # Skip internal/metadata predicates
        if any(skip in pred_lower for skip in ('fact_subject', 'fact_predicate', 'fact_object', 'has_details', 'source_document', 'uploaded_at')):
            continue
        rel_key = (subject_str.lower(), pred_lower, object_str.lower())
        if rel_key in seen:
            continue
        seen.add(rel_key)
        details = get_fact_details(subject_str, predicate_str, object_str)
        source_docs = get_fact_source_document(subject_str, predicate_str, object_str)
        source_doc = source_docs[0][0] if source_docs else "unknown"
        relationships.append({
            'source': subject_str,
            'target': object_str,
            'relationship': predicate_str,
            'details': details or f"{subject_str} {predicate_str} {object_str}",
            'confidence': 0.4,
            'source_document': source_doc,
        })
    return relationships


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
#
# How the causal graph is constructed from data (CSV):
#
# 1. Load CSV: Try multiple delimiters (comma, semicolon, tab) and decimal
#    separators so different formats work. Require at least 2 columns.
#
# 2. Numeric subset: Keep only columns with at least 2 valid numeric values;
#    drop rows/cols that are all NaN. Require at least 10 rows after dropna.
#
# 3. Run PC algorithm (causal-learn): pc(data, alpha=0.05, indep_test=fisherz).
#    Input: (n_samples, n_vars) float64 matrix. Output: causal-learn graph
#    with adjacency matrix G where (i,j)=1 tail at i, (i,j)=-1 head at j
#    => directed edge i -> j.
#
# 4. Interpret adjacency matrix: For each (i,j) with i != j:
#    - (i,j)=-1 and (j,i)=1 => directed edge i -> j (cause -> effect).
#    - (i,j)=1 and (j,i)=-1 => directed edge j -> i.
#    - Other non-zero (undirected/circle) => pick one direction for display.
#
# 5. Enforce DAG: _make_acyclic() keeps only edges that do not create cycles
#    (add edge (u,v) only if there is no path v -> u).
#
# 6. Build UI format: Nodes = one per variable (id, label, type="variable",
#    connections). Edges = list of {id, source, target, label="causes",
#    details, confidence=0.8, source_document="data"}.
#
# 7. Persist for DoWhy: Store data_columns (variable names) and data_rows
#    (up to 5000 rows) so DoWhy can run effect estimation later.
#
# Entry point: discover_causal_structure_from_csv(csv_path, ...).
# Storage: save_data_driven_causal_graph(document_name, graph_data).
# ---------------------------------------------------------------------------

# Edge type constants in causal-learn graph matrix
# 0 = no edge, 1 = tail (arrow from), -1 = head (arrow to), 2 = circle (undirected)
_EDGE_TAIL = 1
_EDGE_HEAD = -1
_EDGE_NONE = 0


def _node_name_to_index(name: str, n_vars: int) -> Optional[int]:
    """Map causal-learn node name (X1, X2, ... 1-based) to 0-based index."""
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if name.startswith("X") and name[1:].isdigit():
        idx = int(name[1:]) - 1
        if 0 <= idx < n_vars:
            return idx
    return None


def _build_edges_list_from_pc_graph(G, n_vars: int) -> List[Tuple[int, int]]:
    """
    Build list of (from_idx, to_idx) from causal-learn graph G using get_graph_edges().
    This is the canonical way to get all relations (directed and undirected).
    - Directed: endpoint1 is TAIL, endpoint2 is ARROW => node1 -> node2.
    - Undirected (CIRCLE-CIRCLE, TAIL-TAIL): show as one directed edge for display.
    """
    edges_list: List[Tuple[int, int]] = []
    try:
        from causallearn.graph.Endpoint import Endpoint
    except ImportError:
        return edges_list
    if not hasattr(G, "get_graph_edges"):
        return edges_list
    graph_edges = G.get_graph_edges()
    if not graph_edges:
        return edges_list
    seen = set()  # (i, j) to avoid duplicate undirected
    for edge in graph_edges:
        n1 = edge.get_node1()
        n2 = edge.get_node2()
        ep1 = edge.get_endpoint1()
        ep2 = edge.get_endpoint2()
        name1 = n1.get_name() if hasattr(n1, "get_name") else str(n1)
        name2 = n2.get_name() if hasattr(n2, "get_name") else str(n2)
        i = _node_name_to_index(name1, n_vars)
        j = _node_name_to_index(name2, n_vars)
        if i is None or j is None or i == j:
            continue
        # Directed: TAIL at node1, ARROW at node2 => 1 -> 2
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            edges_list.append((i, j))
            continue
        if ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            edges_list.append((j, i))
            continue
        # Undirected (CIRCLE-CIRCLE, TAIL-TAIL): show as one directed edge
        if (i, j) not in seen and (j, i) not in seen:
            edges_list.append((i, j))
            seen.add((i, j))
            seen.add((j, i))
    return edges_list


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
        # Keep all columns so every column can form a node; causal discovery uses only numeric subset
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        all_columns = list(df.columns)
        # Numeric columns (for causal discovery): at least 2 valid numbers
        numeric_cols = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 2:
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
        var_names = list(numeric_cols)

        # Drop constant columns (zero variance) to avoid singular correlation matrix in PC
        keep_cols = []
        for j in range(n_vars):
            col = data[:, j]
            if np.var(col) > 1e-10:
                keep_cols.append(j)
        if len(keep_cols) < 2:
            result["error"] = "Need at least 2 non-constant numeric columns (some columns have no variance)"
            return result
        if len(keep_cols) < n_vars:
            data = data[:, keep_cols]
            var_names = [var_names[j] for j in keep_cols]
            n_vars = data.shape[1]

        # Run PC algorithm (causal-learn)
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz
        except ImportError as e:
            result["error"] = "causal-learn is not installed. Install with: pip install causal-learn (or pip install -r requirements-ml.txt)"
            return result

        try:
            cg = pc(data, alpha=alpha, indep_test=fisherz, show_progress=False, stable=True)
        except Exception as pc_err:
            # PC failed (e.g. singular correlation matrix): try to derive relations from data values
            err_msg = str(pc_err)
            conn_for_col = {var_names[i]: 0 for i in range(n_vars)}
            result["nodes"] = [{"id": f"node_{_sanitize_id(col)}", "label": col, "type": "variable", "connections": conn_for_col.get(col, 0)} for col in all_columns]
            result["variable_names"] = var_names
            result["data_columns"] = var_names
            _MAX_DOWHY_ROWS = 5000
            result["data_rows"] = data[: min(len(data), _MAX_DOWHY_ROWS)].tolist()
            edges_list = _derive_edges_from_data(data, var_names, correlation_threshold=0.2)
            if edges_list:
                result["warning"] = f"PC could not run ({err_msg[:80]}…). Relations derived from data (LiNGAM or correlation)."
                node_connections = {i: 0 for i in range(n_vars)}
                for i, j in edges_list:
                    node_connections[i] = node_connections.get(i, 0) + 1
                    node_connections[j] = node_connections.get(j, 0) + 1
                conn_for_col = {var_names[i]: node_connections.get(i, 0) for i in range(n_vars)}
                result["nodes"] = [{"id": f"node_{_sanitize_id(col)}", "label": col, "type": "variable", "connections": conn_for_col.get(col, 0)} for col in all_columns]
                result["edges"] = [{"id": f"edge_{idx}", "source": f"node_{_sanitize_id(var_names[i])}", "target": f"node_{_sanitize_id(var_names[j])}", "label": "causes", "details": f"Data-driven: {var_names[i]} → {var_names[j]}", "confidence": 0.8, "is_inferred": False, "source_document": "data"} for idx, (i, j) in enumerate(edges_list)]
                result["stats"] = {"total_nodes": len(all_columns), "total_edges": len(edges_list), "direct_relationships": len(edges_list), "inferred_relationships": 0, "most_causal_entities": [{"entity": var_names[i], "connections": node_connections.get(i, 0)} for i in sorted(range(n_vars), key=lambda k: node_connections.get(k, 0), reverse=True)[:10]], "algorithm": "data-derived", "source": "data"}
                result["dowhy_pairs"] = [(var_names[i], var_names[j]) for i, j in edges_list]
            else:
                result["warning"] = f"Causal discovery could not run: {err_msg}. Showing variable names only."
                result["edges"] = []
                result["stats"] = {"total_nodes": len(all_columns), "total_edges": 0, "direct_relationships": 0, "inferred_relationships": 0, "most_causal_entities": [], "algorithm": "PC", "source": "data"}
                result["dowhy_pairs"] = []
            return result

        # Build edges from causal-learn graph: use get_graph_edges() so all relations (directed + undirected) are shown
        g = cg.G.graph
        if g is None:
            result["error"] = "PC returned no graph"
            result["nodes"] = [{"id": f"node_{_sanitize_id(col)}", "label": col, "type": "variable", "connections": 0} for col in all_columns]
            result["stats"]["total_nodes"] = len(all_columns)
            return result

        edges_list = _build_edges_list_from_pc_graph(cg.G, n_vars)
        # Fallback: if get_graph_edges gave nothing, parse adjacency matrix (causal-learn matrix: 1=tail, -1=head)
        if not edges_list:
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        continue
                    val_ij = int(g[i, j]) if hasattr(g, "__getitem__") else 0
                    val_ji = int(g[j, i]) if hasattr(g, "__getitem__") else 0
                    if val_ij == _EDGE_HEAD and val_ji == _EDGE_TAIL:
                        edges_list.append((i, j))
                    elif val_ij == _EDGE_TAIL and val_ji == _EDGE_HEAD:
                        edges_list.append((j, i))
                    elif val_ij != _EDGE_NONE or val_ji != _EDGE_NONE:
                        if (j, i) not in edges_list and (i, j) not in edges_list:
                            edges_list.append((i, j))

        # Ensure acyclic for display (directed edges from PC are usually acyclic; this avoids cycles from undirected)
        edges_list = _make_acyclic(edges_list, n_vars)

        # If PC gave no edges, derive relations from data (LiNGAM or correlation)
        if not edges_list:
            edges_list = _derive_edges_from_data(data, var_names, correlation_threshold=0.2)
            if edges_list:
                result["warning"] = "PC found no edges; relations derived from data (LiNGAM or correlation)."

        node_connections = {i: 0 for i in range(n_vars)}
        for i, j in edges_list:
            node_connections[i] = node_connections.get(i, 0) + 1
            node_connections[j] = node_connections.get(j, 0) + 1

        # All columns form nodes; connection counts from discovery variables (var_names)
        conn_for_col = {var_names[i]: node_connections.get(i, 0) for i in range(n_vars)}
        nodes = []
        for col in all_columns:
            node_id = f"node_{_sanitize_id(col)}"
            nodes.append({
                "id": node_id,
                "label": col,
                "type": "variable",
                "connections": conn_for_col.get(col, 0),
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
        # 2. Extract candidate (treatment, outcome) pairs from directed edges for DoWhy
        result["dowhy_pairs"] = _extract_directed_pairs_from_pc_graph(cg.G, var_names)
        # Persist numeric data for DoWhy (use same columns as graph; data may be subset if constant cols were dropped)
        _MAX_DOWHY_ROWS = 5000
        result["data_columns"] = var_names
        result["data_rows"] = data[: min(len(data), _MAX_DOWHY_ROWS)].tolist()
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def _extract_directed_pairs_from_pc_graph(G, var_names: List[str]) -> List[Tuple[str, str]]:
    """
    From causal-learn graph G (from PC), extract (treatment, outcome) pairs for each directed edge.
    Node names in causal-learn are typically "X1", "X2", ... (1-based) -> map to var_names[index].
    """
    pairs: List[Tuple[str, str]] = []
    try:
        from causallearn.graph.Endpoint import Endpoint
    except ImportError:
        return pairs
    if not hasattr(G, "get_graph_edges"):
        return pairs
    edges = G.get_graph_edges()
    if not edges:
        return pairs
    n_vars = len(var_names)
    # X1 -> index 0, X2 -> index 1, ...
    def node_name_to_var(name: str) -> Optional[str]:
        if not name or not isinstance(name, str):
            return None
        name = name.strip()
        if name.startswith("X") and name[1:].isdigit():
            idx = int(name[1:]) - 1
            if 0 <= idx < n_vars:
                return var_names[idx]
        return None
    for edge in edges:
        ep1, ep2 = edge.get_endpoint1(), edge.get_endpoint2()
        if ep1 != Endpoint.TAIL or ep2 != Endpoint.ARROW:
            continue
        n1 = edge.get_node1()
        n2 = edge.get_node2()
        name1 = n1.get_name() if hasattr(n1, "get_name") else str(n1)
        name2 = n2.get_name() if hasattr(n2, "get_name") else str(n2)
        t = node_name_to_var(name1)
        y = node_name_to_var(name2)
        if t and y and t != y:
            pairs.append((t, y))
    return pairs


def _derive_edges_from_data(
    data: Any,
    var_names: List[str],
    correlation_threshold: float = 0.2,
) -> List[Tuple[int, int]]:
    """
    When PC gives no edges, derive relations from data values:
    1. Try LiNGAM (directed edges from linear non-Gaussian model).
    2. Else use correlation: add edge (i, j) for |corr(i,j)| >= threshold (direction i->j for i<j).
    Returns list of (from_idx, to_idx).
    """
    import numpy as np
    n_vars = len(var_names)
    if n_vars < 2 or data is None or data.shape[1] < 2:
        return []

    # 1. Try LiNGAM (directed edges)
    try:
        from causallearn.search.FCMBased.lingam import DirectLiNGAM
        model = DirectLiNGAM()
        model.fit(np.asarray(data, dtype=np.float64))
        B = model.adjacency_matrix_
        if B is not None:
            n = min(B.shape[0], n_vars)
            edges = []
            for i in range(n):
                for j in range(n):
                    if i != j and B[i, j] != 0:
                        edges.append((j, i))  # LiNGAM: B[i,j] => j -> i
            if edges:
                return _make_acyclic(edges, n_vars)
    except Exception:
        pass

    # 2. Correlation-based: edge between variables with |corr| >= threshold
    try:
        corr = np.corrcoef(data.T)
        if corr is None or corr.shape[0] < 2:
            return []
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if i < corr.shape[0] and j < corr.shape[1]:
                    r = corr[i, j]
                    if np.isfinite(r) and abs(r) >= correlation_threshold:
                        edges.append((i, j))  # one direction for display
        return edges
    except Exception:
        return []


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
    if graph_data.get("dowhy_pairs") is not None:
        entry["dowhy_pairs"] = graph_data["dowhy_pairs"]
    if graph_data.get("warning"):
        entry["warning"] = graph_data["warning"]
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
        "interpretation": None,
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
        data_ctx = None
        if treatment in df.columns and outcome in df.columns:
            try:
                data_ctx = {
                    "treatment_min": float(df[treatment].min()),
                    "treatment_max": float(df[treatment].max()),
                    "outcome_min": float(df[outcome].min()),
                    "outcome_max": float(df[outcome].max()),
                    "n_rows": len(df),
                }
            except Exception:
                pass
        out["interpretation"] = _format_effect_interpretation(
            treatment, outcome, out["estimate_value"], data_context=data_ctx
        )
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


def _format_effect_interpretation(
    treatment: str,
    outcome: str,
    estimate_value: Optional[float],
    data_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a plain-language interpretation of the causal effect, with data context when available."""
    if estimate_value is None:
        return "No numeric effect was estimated."
    t, o = treatment.replace("_", " "), outcome.replace("_", " ")
    val = estimate_value
    abs_val = abs(val)
    data_line = ""
    if data_context:
        n = data_context.get("n_rows")
        tx_min = data_context.get("treatment_min")
        tx_max = data_context.get("treatment_max")
        o_min = data_context.get("outcome_min")
        o_max = data_context.get("outcome_max")
        parts = []
        if n is not None:
            parts.append(f"Based on {n} rows in the data")
        if tx_min is not None and tx_max is not None:
            parts.append(f"{t!r} ranges from {tx_min:.2g} to {tx_max:.2g}")
        if o_min is not None and o_max is not None:
            parts.append(f"{o!r} from {o_min:.2g} to {o_max:.2g}")
        if parts:
            data_line = " ".join(parts) + ". "
    if abs_val < 1e-6:
        return (
            f"{data_line}Effect near zero: little or no linear causal effect of {t!r} on {o!r} in this model. "
            "Changes in the treatment are not associated with systematic changes in the outcome."
        )
    if val > 0:
        return (
            f"{data_line}Positive effect: an increase in {t!r} tends to increase {o!r}. "
            f"On average, a one-unit increase in the treatment is associated with an increase of about {abs_val:.4g} units in the outcome."
        )
    return (
        f"{data_line}Negative effect: an increase in {t!r} tends to decrease {o!r}. "
        f"On average, a one-unit increase in the treatment is associated with a decrease of about {abs_val:.4g} units in the outcome."
    )


def run_dowhy_one_pair(
    graph_data: Dict[str, Any],
    treatment: str,
    outcome: str,
    method: str = "backdoor.linear_regression",
) -> Dict[str, Any]:
    """
    Run DoWhy causal effect estimation for one (treatment, outcome) pair using
    graph_data (no store). graph_data must have nodes, edges, data_columns, data_rows.
    """
    import pandas as pd

    out = {
        "success": False,
        "treatment": treatment,
        "outcome": outcome,
        "estimate": None,
        "estimate_value": None,
        "interpretation": None,
        "refutation": None,
        "error": None,
    }
    try:
        cols = graph_data.get("data_columns")
        rows = graph_data.get("data_rows")
        if not cols or rows is None:
            out["error"] = "No data in graph_data (data_columns / data_rows)"
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
        model = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=nx_graph)
        identified = model.identify_effect()
        estimate = model.estimate_effect(identified, method_name=method)
        out["estimate_value"] = float(estimate.value) if estimate.value is not None else None
        out["estimate"] = str(estimate)
        data_ctx = None
        if treatment in df.columns and outcome in df.columns:
            try:
                data_ctx = {
                    "treatment_min": float(df[treatment].min()),
                    "treatment_max": float(df[treatment].max()),
                    "outcome_min": float(df[outcome].min()),
                    "outcome_max": float(df[outcome].max()),
                    "n_rows": len(df),
                }
            except Exception:
                pass
        out["interpretation"] = _format_effect_interpretation(
            treatment, outcome, out["estimate_value"], data_context=data_ctx
        )
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


def run_dowhy_for_all_edges(
    graph_data: Dict[str, Any],
    method: str = "backdoor.linear_regression",
) -> Dict[str, Any]:
    """
    Run DoWhy for each directed edge (treatment, outcome) in the discovered graph.
    Uses graph_data["dowhy_pairs"] if present, else extracts pairs from graph_data["edges"].
    Returns {"pairs": [(t,y), ...], "results": {(t,y): {estimate_value, estimate, refutation, ...}}}.
    """
    pairs = list(graph_data.get("dowhy_pairs") or [])
    if not pairs:
        node_id_to_label = {n["id"]: n["label"] for n in graph_data.get("nodes", [])}
        for e in graph_data.get("edges", []):
            src = node_id_to_label.get(e["source"], e["source"].replace("node_", "").replace("_", " "))
            tgt = node_id_to_label.get(e["target"], e["target"].replace("node_", "").replace("_", " "))
            if src and tgt and src != tgt:
                pairs.append((src, tgt))
    results = {}
    for (t, y) in pairs:
        one = run_dowhy_one_pair(graph_data, t, y, method=method)
        results[(t, y)] = one
    return {"pairs": pairs, "results": results}


def run_direct_lingam(data: Any, var_names: List[str]) -> List[Tuple[str, str]]:
    """
    Run DirectLiNGAM on data to get directed edges (optional direction refinement).
    data: (n_samples, n_vars) array; var_names: list of variable names.
    Returns list of (cause, effect) as variable names.
    """
    try:
        from causallearn.search.FCMBased.lingam import DirectLiNGAM
    except ImportError:
        return []
    try:
        model = DirectLiNGAM()
        model.fit(data)
        # LiNGAM: x = Bx + e, so B[i,j] = effect of x_j on x_i => edge j -> i
        B = model.adjacency_matrix_
        if B is None:
            return []
        n = min(B.shape[0], len(var_names))
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and B[i, j] != 0:
                    edges.append((var_names[j], var_names[i]))  # j -> i
        return edges
    except Exception:
        return []


def list_data_driven_causal_graph_sources() -> List[str]:
    """List document names that have a stored data-driven causal graph."""
    store = _load_causal_graphs_store()
    names = list(store.get("graphs", {}).keys())
    if names:
        print(f"📊 Causal graph store: {_get_store_path()} → {len(names)} dataset(s): {names[:5]}{'...' if len(names) > 5 else ''}")
    return names


def remove_data_driven_causal_graph(document_name: str) -> bool:
    """Remove the data-driven causal graph for a document (e.g. when document is deleted). Returns True if removed."""
    store = _load_causal_graphs_store()
    graphs = store.get("graphs", {})
    if document_name in graphs:
        del graphs[document_name]
        _save_causal_graphs_store(store)
        print(f"🧹 Removed causal graph for document: {document_name}")
        return True
    return False


def clear_data_driven_causal_graphs() -> None:
    """Clear all data-driven causal graphs (call on backend startup).
    After restart, only CSVs uploaded in the current session will appear in the causal graph source list."""
    store = {"graphs": {}, "last_updated": None}
    _save_causal_graphs_store(store)
    path = _get_store_path()
    print(f"🧹 Causal graph store cleared at {path} — only datasets uploaded this session will appear")


def get_causal_graph_data(include_inferred: bool = True, source_document_filter: Optional[str] = None) -> Dict:
    """
    Get causal graph data in format suitable for visualization.

    Args:
        include_inferred: Whether to include inferred causal relationships
        source_document_filter: If set, only include relationships from this document (for PDF/DOCX/TXT).

    Returns:
        Dictionary with nodes, edges, stats.
    """
    # Extract direct causal relationships (predicates that match causal keywords)
    causal_rels = extract_causal_relationships()

    # If no causal predicates matched but the KB has facts, show all relationships as weak causal
    if len(causal_rels) == 0 and len(graph) > 0:
        causal_rels = extract_all_relationships_as_weak_causal()

    if source_document_filter:
        causal_rels = [r for r in causal_rels if r.get("source_document") == source_document_filter]

    # Optionally infer additional relationships (inferred from the filtered causal set)
    inferred_rels = []
    if include_inferred and causal_rels:
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
