"""LAYA-SWE: Fast System-One Decision Models for Autonomous Coding Agent Memory.

Organizes agent observations into four discrete execution planes:
1. INVARIANT Plane: Pinned architectural constraints, security rules, and negative invariants (immune from LRU eviction).
2. SYMBOLIC Plane: Structural AST outlines, symbol dependencies, and module signatures.
3. TRAJECTORY Plane: Shell execution logs, test outputs, tracebacks, and tool calls.
4. RESOLUTION Plane: Causal bug resolutions, passing test patches, and verified fixes.
"""

from enum import Enum
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None


def bound_state_tokens(text: str, max_tokens: int = 300) -> str:
    """Explicitly slice state to fit the Laya English checkpoint's 512-token sequence limit (~320 token state budget)."""
    if not text:
        return ""
    if _enc is not None:
        try:
            tokens = _enc.encode(text)
            if len(tokens) > max_tokens:
                half = max_tokens // 2
                head = _enc.decode(tokens[:half])
                tail = _enc.decode(tokens[-half:])
                return f"{head}\n... [truncated] ...\n{tail}"
            return text
        except Exception:
            pass
    # Conservative character fallback: 3 chars per token on dense code/tracebacks
    max_chars = max_tokens * 3
    if len(text) > max_chars:
        half = max_chars // 2
        return f"{text[:half]}\n... [truncated] ...\n{text[-half:]}"
    return text


logger = logging.getLogger("laya_swe_memory_engine")
logging.basicConfig(level=logging.INFO)


class MemoryPlane(str, Enum):
    INVARIANT = "invariant"     # Architectural negative constraints, security rules, bounds
    SYMBOLIC = "symbolic"       # File structures, AST class/function signatures, exports
    TRAJECTORY = "trajectory"   # Shell commands, execution logs, test outputs, tracebacks
    RESOLUTION = "resolution"   # Causal bug fixes, verified hypotheses, passing patches


@dataclass
class MemoryNode:
    node_id: str
    content: str
    plane: MemoryPlane
    timestamp: float
    symbols: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    is_pinned: bool = False     # Pinned invariants are never evicted by LRU
    last_accessed: float = 0.0  # Unix timestamp for true LRU eviction
    role: str = "tool"          # Provenance origin role (system, user, tool, assistant)

    def __post_init__(self):
        if not self.last_accessed:
            self.last_accessed = self.timestamp

    @property
    def memory_type(self) -> str:
        """Backward-compatibility mapping for existing test assertions."""
        if self.plane == MemoryPlane.INVARIANT:
            return "constraint"
        elif self.plane == MemoryPlane.SYMBOLIC:
            return "semantic"
        elif self.plane == MemoryPlane.RESOLUTION:
            return "preference"
        else:
            return "episodic"


# Backward compatibility aliases
AxiomNode = MemoryNode


class LayaMemoryController:
    """Multi-Mode SWE Memory Controller supporting LAYA Neural Model and Deterministic Structural Engine."""

    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or os.getenv("LAYA_CONTROLLER_MODE") or os.getenv("AXIOM_CONTROLLER_MODE", "laya_hybrid")
        self._laya_agent = None
        self._laya_init_failed = False

    def get_laya_agent(self):
        """Lazy singleton loader for local LAYA System-1 Agent."""
        if self._laya_agent is None and not self._laya_init_failed:
            try:
                import laya
                logger.info("Initializing LAYA System-1 Decision Agent (convaiinnovations/laya)...")
                t0 = time.time()
                self._laya_agent = laya.load("convaiinnovations/laya", device="cpu")
                logger.info(f"LAYA System-1 Decision Agent ready in {time.time() - t0:.2f}s")
            except Exception as e:
                logger.warning(f"Failed to load LAYA agent: {e}. Falling back to deterministic mode.")
                self._laya_init_failed = True
        return self._laya_agent

    def extract_symbols(self, text: str) -> List[str]:
        """Extract code symbols, file paths, and test identifiers deterministically."""
        symbols: Set[str] = set()

        # File paths (e.g. app/analytics.py, tests/test_tenant.py)
        for match in re.findall(r"\b(?:[a-zA-Z0-9_\-]+/)+[a-zA-Z0-9_\-]+\.(?:py|ts|js|json|yaml|yml|md|txt)\b", text):
            symbols.add(match)

        # Python class and function definitions
        for match in re.findall(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            symbols.add(match)

        # Test case identifiers
        for match in re.findall(r"\btest_[A-Za-z0-9_]+\b", text):
            symbols.add(match)

        # Architectural identifiers & keywords
        keywords = ("tenant_id", "hmac", "sha256", "lru", "coordinator", "ledger", "ast", "2pc", "abort", "commit")
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                symbols.add(kw)

        return sorted(list(symbols))[:12]

    def classify_plane(self, content: str, metadata: Optional[Dict] = None) -> Tuple[MemoryPlane, bool]:
        """Classify observation into an SWE memory plane and determine if it should be pinned.

        Supports:
        - 'laya_hybrid': LAYA neural prediction with deterministic invariant safety bounds (Default).
        - 'laya_pure': Pure LAYA neural decision model classification.
        - 'deterministic': Pure heuristic regex and AST pattern matching.
        """
        meta = metadata if metadata is not None else {}
        content_lower = content.lower()

        laya_info = {
            "mode": self.mode,
            "choice": None,
            "probabilities": {},
            "confidence": 0.0,
            "latency": 0.0
        }

        # 1. Neural Classification via real LAYA System-1 Agent
        if self.mode in ("laya_pure", "laya_hybrid") and not self._laya_init_failed:
            agent = self.get_laya_agent()
            if agent:
                try:
                    t_start = time.time()
                    questions = {
                        "swe_triage": {
                            "type": "choice",
                            "instructions": "Classify this software engineering text into the appropriate category.",
                            "criteria": {
                                "invariant": "A strict rule, security constraint, or negative instruction that must never be broken",
                                "failure": "A test failure, exception, crash trace, or assertion error",
                                "code": "Function definition, class implementation, or source code",
                                "log": "General build, execution, or progress logging output"
                            }
                        }
                    }
                    # The Laya English checkpoint (convaiinnovations/laya) has a 512-token sequence limit (~320 tokens for state).
                    # We bound state tokens accurately via BPE tokenizer / character fallback:
                    bounded_state = bound_state_tokens(content, max_tokens=300)

                    res = agent.predict(state=bounded_state, questions=questions)
                    t_elapsed = time.time() - t_start

                    ans = res.get("answers", {}).get("swe_triage", {})
                    choice = ans.get("choice")
                    probs = ans.get("probabilities", {})
                    conf = ans.get("confidence", 0.0)

                    laya_info["choice"] = choice
                    laya_info["probabilities"] = probs
                    laya_info["confidence"] = conf
                    laya_info["latency"] = round(t_elapsed, 4)
                    meta["laya"] = laya_info

                    if self.mode == "laya_pure":
                        if choice == "invariant":
                            return MemoryPlane.INVARIANT, True
                        elif choice == "failure":
                            is_res = ("0 failed" in content_lower or "passed in" in content_lower)
                            return (MemoryPlane.RESOLUTION if is_res else MemoryPlane.TRAJECTORY), False
                        elif choice == "code":
                            return MemoryPlane.SYMBOLIC, False
                        else:
                            return MemoryPlane.TRAJECTORY, False

                    elif self.mode == "laya_hybrid":
                        is_neural_inv = (choice == "invariant" and probs.get("invariant", 0.0) >= 0.55)
                        is_heur_inv = any(phrase in content_lower for phrase in (
                            "must not", "never", "do not", "mandatory", "invariant",
                            "security violation", "standard library only", "no third-party", "prohibited"
                        )) or meta.get("type") == "invariant"

                        if is_neural_inv or is_heur_inv:
                            return MemoryPlane.INVARIANT, True
                        if choice == "failure" and not ("0 failed" in content_lower or "passed in" in content_lower):
                            return MemoryPlane.TRAJECTORY, False
                        if choice == "code":
                            return MemoryPlane.SYMBOLIC, False
                        if any(phrase in content_lower for phrase in ("passed in", "all tests pass", "fix verified")):
                            return MemoryPlane.RESOLUTION, False
                        return MemoryPlane.TRAJECTORY, False

                except Exception as e:
                    logger.warning(f"LAYA neural inference error: {e}. Falling back to deterministic.")

        # 2. Deterministic Fallback Mode
        is_invariant = any(phrase in content_lower for phrase in (
            "must not", "never", "do not", "mandatory", "invariant",
            "security violation", "standard library only", "no third-party", "prohibited"
        )) or meta.get("type") == "invariant"

        if is_invariant:
            return MemoryPlane.INVARIANT, True

        is_resolution = any(phrase in content_lower for phrase in (
            "passed in", "all tests pass", "fix verified", "bug resolved", "100% passing"
        )) and ("failed" not in content_lower or "0 failed" in content_lower)
        if is_resolution:
            return MemoryPlane.RESOLUTION, False

        is_symbolic = any(line.strip().startswith(("class ", "def ", "import ", "from ", "@dataclass", "interface ")) for line in content.splitlines())
        if is_symbolic:
            return MemoryPlane.SYMBOLIC, False

        return MemoryPlane.TRAJECTORY, False

    def classify_and_score(self, content: str) -> Dict[str, float]:
        """Continuous plane scoring for backwards compatibility with SystemOneController."""
        plane, is_pinned = self.classify_plane(content)
        scores = {"constraint": 0.1, "procedural": 0.1, "preference": 0.1, "episodic": 0.1, "semantic": 0.1}

        if plane == MemoryPlane.INVARIANT:
            scores["constraint"] = 0.95
        elif plane == MemoryPlane.RESOLUTION:
            scores["preference"] = 0.85
            scores["episodic"] = 0.70
        elif plane == MemoryPlane.SYMBOLIC:
            scores["semantic"] = 0.90
        else:
            scores["episodic"] = 0.90
            scores["procedural"] = 0.70
        return scores

    def judge_relation(self, source_node: MemoryNode, target_node: MemoryNode) -> Dict[str, float]:
        """Judge structural, causal, and invariant relationships between two nodes."""
        s_syms = set(source_node.symbols)
        t_syms = set(target_node.symbols)
        sym_overlap = len(s_syms & t_syms) / max(1, len(s_syms | t_syms))

        s_words = set(re.findall(r"[A-Za-z0-9_]{3,}", source_node.content.lower()))
        t_words = set(re.findall(r"[A-Za-z0-9_]{3,}", target_node.content.lower()))
        shared_code_terms = bool(s_words & t_words & {"tenant", "session", "db", "query", "cache", "lru", "lock", "tx", "commit", "abort", "ast", "requirements", "stdlib"})

        # Check invariant guard relation: invariant + shared symbols/identifiers
        is_guard = (source_node.plane == MemoryPlane.INVARIANT or target_node.plane == MemoryPlane.INVARIANT) and (bool(s_syms & t_syms) or shared_code_terms)

        # Check causal failure-fix relation:
        # Requires:
        # 1. One node is TRAJECTORY and the other is RESOLUTION
        # 2. Shared code symbols, module paths, or test/error keywords
        is_traj_res = (source_node.plane == MemoryPlane.TRAJECTORY and target_node.plane == MemoryPlane.RESOLUTION) or \
                      (target_node.plane == MemoryPlane.TRAJECTORY and source_node.plane == MemoryPlane.RESOLUTION)
        shared_err_terms = bool(s_words & t_words & {"eviction", "deadlock", "race", "assert", "assertion", "error", "failed", "test", "fail", "fix", "pass", "lock", "lru", "patch"})
        is_causal = is_traj_res and (bool(s_syms & t_syms) or shared_err_terms or len(s_words & t_words) >= 4)

        return {
            "symbolic": min(1.0, sym_overlap * 2.5),
            "guard": 0.95 if is_guard else 0.0,
            "causal": 0.90 if is_causal else 0.0,
            "semantic": min(1.0, sym_overlap * 2.0)
        }

    def assess_evidence_sufficiency(self, query: str, retrieved_nodes: List[MemoryNode]) -> Tuple[bool, float]:
        """Deterministic stopping check based on symbol and invariant coverage."""
        if not retrieved_nodes:
            return False, 0.0

        q_symbols = set(self.extract_symbols(query))
        if not q_symbols:
            q_symbols = set(re.findall(r"\w+", query.lower()))

        covered_symbols = set()
        has_invariant = False
        for node in retrieved_nodes:
            covered_symbols.update(node.symbols)
            covered_symbols.update(re.findall(r"\w+", node.content.lower()))
            if node.plane == MemoryPlane.INVARIANT:
                has_invariant = True

        coverage = len(q_symbols & covered_symbols) / max(1, len(q_symbols))
        is_sufficient = (coverage >= 0.60) or (has_invariant and len(retrieved_nodes) >= 2)
        return is_sufficient, coverage


# Backward compatibility aliases
AxiomMemoryController = LayaMemoryController
SystemOneController = LayaMemoryController


class LayaSweMemoryEngine:
    """Multi-Plane SWE Memory Engine for Autonomous Coding Agents."""

    def __init__(self, controller: Optional[LayaMemoryController] = None, max_nodes: int = 250, max_pinned: int = 15):
        self.controller = controller or LayaMemoryController()
        self.max_nodes = max_nodes
        self.max_pinned = max_pinned
        self.graph = nx.MultiDiGraph()
        self.nodes_map: Dict[str, MemoryNode] = {}
        self.node_counter = 0
        self._lock = threading.Lock()

    def revoke_invariant(self, pattern_or_symbol: str) -> int:
        """Revoke or unpin an invariant if a user explicitly relaxes a rule (Invariant Lifecycle)."""
        revoked = 0
        p_lower = pattern_or_symbol.lower().strip()
        with self._lock:
            for node in self.nodes_map.values():
                if node.is_pinned:
                    if p_lower in node.content.lower() or any(p_lower == s.lower() for s in node.symbols):
                        node.is_pinned = False
                        revoked += 1
                        logger.info(f"[LAYA-SWE Security] Revoked pinned invariant on node {node.node_id}: {node.content[:60]}")
        return revoked

    def ingest(
        self,
        observation: str,
        plane: Optional[MemoryPlane] = None,
        metadata: Optional[Dict] = None,
        role: Optional[str] = None
    ) -> str:
        """Ingest a new conversational or tool observation into the memory graph with provenance gating."""
        text = observation.strip()
        if not text or len(text) < 5:
            return ""

        metadata = metadata or {}
        # Determine provenance role: explicit role arg > metadata["role"] > default "user"
        origin_role = (role or metadata.get("role") or "user").lower()

        # Check for supersession / constraint relaxation directives in user input (Invariant Lifecycle)
        if origin_role in ("system", "user", "human"):
            relax_keywords = ("you may now", "allow ", "override ", "relax ", "no longer need", "permission to", "remove invariant")
            if any(kw in text.lower() for kw in relax_keywords):
                with self._lock:
                    for node in list(self.nodes_map.values()):
                        if node.is_pinned:
                            inv_words = set(re.findall(r"[a-zA-Z0-9_\-\.]{4,}", node.content.lower())) - {
                                "mandatory", "invariant", "under", "circumstance", "never", "always", "must", "with"
                            }
                            text_words = set(re.findall(r"[a-zA-Z0-9_\-\.]{4,}", text.lower()))
                            if (set(node.symbols) & text_words) or (inv_words & text_words):
                                node.is_pinned = False
                                logger.info(f"[LAYA-SWE Security] Superseded/revoked pinned invariant on node {node.node_id}: {node.content[:60]}")

        # 1. Plane Classification
        if plane is None:
            detected_plane, candidate_pinned = self.controller.classify_plane(text, metadata)
        else:
            detected_plane = plane
            candidate_pinned = (plane == MemoryPlane.INVARIANT)

        # 2. Provenance-Gated Pinning Security ("When Context Gets Root" Mitigation)
        # Strict Rule: Untrusted tool output can NEVER be pinned as an invariant,
        # preventing privilege escalation via adversarial tool results.
        if origin_role not in ("system", "user", "human"):
            is_pinned = False
            if detected_plane == MemoryPlane.INVARIANT:
                detected_plane = MemoryPlane.TRAJECTORY
                text = f"[OBSERVED TOOL OUTPUT] {text}"
        else:
            is_pinned = candidate_pinned

        # 3. Invariant Deduplication & Hard Pin Cap
        with self._lock:
            if is_pinned:
                norm_text = re.sub(r"\s+", " ", text.lower().strip())
                for existing_node in self.nodes_map.values():
                    if existing_node.is_pinned:
                        existing_norm = re.sub(r"\s+", " ", existing_node.content.lower().strip())
                        if norm_text == existing_norm or norm_text in existing_norm:
                            existing_node.last_accessed = time.time()
                            return existing_node.node_id

                current_pinned = [n for n in self.nodes_map.values() if n.is_pinned]
                if len(current_pinned) >= self.max_pinned:
                    is_pinned = False
                    logger.warning(f"[LAYA-SWE Security] Pinned invariant cap ({self.max_pinned}) reached; ingesting as unpinned.")

            self.node_counter += 1
            node_id = f"mem_{self.node_counter:04d}"

            # 4. Symbol Extraction
            symbols = metadata.get("symbols") or self.controller.extract_symbols(text)
            now = time.time()
            node = MemoryNode(
                node_id=node_id,
                content=text,
                plane=detected_plane,
                timestamp=now,
                symbols=symbols,
                metadata=metadata,
                is_pinned=is_pinned,
                last_accessed=now,
                role=origin_role
            )
            self.nodes_map[node_id] = node
            self.graph.add_node(node_id, node=node)

            # 5. Structural & Invariant Graph Wiring
            candidates = list(self.nodes_map.values())[-12:-1]
            for candidate in candidates:
                relations = self.controller.judge_relation(node, candidate)
                for rel_type, score in relations.items():
                    if score >= 0.5:
                        self.graph.add_edge(node.node_id, candidate.node_id, rel_type=rel_type, weight=score)
                        self.graph.add_edge(candidate.node_id, node.node_id, rel_type=rel_type, weight=score)

                # Temporal sequencing
                self.graph.add_edge(candidate.node_id, node.node_id, rel_type="temporal_precedes", weight=1.0)

            # 6. True LRU Eviction (access-time weighted)
            if len(self.nodes_map) > self.max_nodes:
                unpinned_candidates = [n for n in self.nodes_map.values() if not n.is_pinned]
                if unpinned_candidates:
                    evict_node = min(unpinned_candidates, key=lambda n: n.last_accessed)
                else:
                    evict_node = min(self.nodes_map.values(), key=lambda n: n.last_accessed)

                del self.nodes_map[evict_node.node_id]
                if self.graph.has_node(evict_node.node_id):
                    self.graph.remove_node(evict_node.node_id)
                logger.debug(f"[LAYA-SWE] Evicted unpinned node {evict_node.node_id} (True LRU cap {self.max_nodes})")

            num_edges = len(self.graph.edges(node_id))

        logger.info(f"[LAYA-SWE] Ingested node {node_id} [plane: {detected_plane.value}, pinned: {is_pinned}, role: {origin_role}, edges: {num_edges}]")
        return node_id

    def retrieve_two_slot(self, query: str, top_k: int = 5) -> Tuple[str, str]:
        """Retrieve memory split into two discrete slots to prevent context-reconstruction privilege escalation:
        1. privileged_invariants: Pinned security and architectural constraints (Provenance: system/user only).
        2. unprivileged_evidence: Delimited, untrusted tool/AST observations tagged as observed data.
        """
        now = time.time()
        with self._lock:
            if not self.nodes_map:
                return "", "No memory records available."
            pinned_invariants = [n for n in self.nodes_map.values() if n.is_pinned]
            unpinned_candidates_list = [n for n in self.nodes_map.values() if not n.is_pinned]

        query_symbols = set(self.controller.extract_symbols(query))
        query_words = set(re.findall(r"\w+", query.lower()))

        # 1. Score unpinned candidates
        scored_candidates = []
        for node in unpinned_candidates_list:
            node_symbols = set(node.symbols)
            node_words = set(re.findall(r"\w+", node.content.lower()))

            sym_score = len(query_symbols & node_symbols) * 4.0
            word_score = len(query_words & node_words) * 1.5

            if sym_score > 0 or word_score > 0:
                plane_boost = 0.0
                if node.plane == MemoryPlane.RESOLUTION:
                    plane_boost = 2.5
                elif node.plane == MemoryPlane.SYMBOLIC:
                    plane_boost = 1.5
                total_score = sym_score + word_score + plane_boost
                scored_candidates.append((node, total_score))

        # Fallback if query had no keyword matches and no pinned invariants exist
        if not scored_candidates and not pinned_invariants:
            scored_candidates = [(node, 0.1) for node in unpinned_candidates_list[-top_k:]]

        scored_candidates.sort(key=lambda x: -x[1])

        # 2. Multi-Relational Graph Edge Traversal
        visited_ids = {n.node_id for n in pinned_invariants}
        traversed_boosted: List[Tuple[MemoryNode, float]] = []

        with self._lock:
            top_seed_nodes = [node for node, score in scored_candidates[:3]]
            for seed_node in top_seed_nodes:
                visited_ids.add(seed_node.node_id)
                if self.graph.has_node(seed_node.node_id):
                    for neighbor_id in self.graph.neighbors(seed_node.node_id):
                        if neighbor_id in visited_ids:
                            continue
                        neighbor = self.nodes_map.get(neighbor_id)
                        if not neighbor:
                            continue
                        edge_data = self.graph.get_edge_data(seed_node.node_id, neighbor_id) or {}
                        for edge in edge_data.values():
                            rel = edge.get("rel_type")
                            if rel == "causal" and neighbor.plane == MemoryPlane.RESOLUTION:
                                traversed_boosted.append((neighbor, 8.0))
                                visited_ids.add(neighbor_id)
                                break
                            elif rel == "guard" and neighbor.plane == MemoryPlane.INVARIANT:
                                traversed_boosted.append((neighbor, 7.0))
                                visited_ids.add(neighbor_id)
                                break

        all_candidates = traversed_boosted + scored_candidates
        all_candidates.sort(key=lambda x: -x[1])

        # 3. Dynamic Evidence Selection with Adaptive Stopping
        selected_unpinned: List[MemoryNode] = []
        for node, score in all_candidates:
            if len(selected_unpinned) >= top_k:
                break
            if node not in selected_unpinned and not node.is_pinned:
                selected_unpinned.append(node)
                sufficient, coverage = self.controller.assess_evidence_sufficiency(query, list(pinned_invariants) + selected_unpinned)
                if sufficient and len(selected_unpinned) >= min(top_k, 3):
                    logger.info(f"[LAYA-SWE] Adaptive stopping triggered ({len(selected_unpinned)} nodes, coverage: {coverage:.2f})")
                    break

        # 4. Refresh LRU Access Time
        with self._lock:
            for n in list(pinned_invariants) + selected_unpinned:
                n.last_accessed = now

        # 5. Format Slot 1: Privileged Invariants (Safe for System Slot)
        inv_lines = []
        for n in pinned_invariants:
            c = n.content.strip()
            if len(c) > 300:
                c = c[:300] + "... [truncated]"
            inv_lines.append(f"• [MANDATORY INVARIANT] {c}")
        privileged_invariants = "\n".join(inv_lines)

        # 6. Format Slot 2: Unprivileged Evidence (Untrusted Tool/AST Data)
        ev_lines = []
        for n in selected_unpinned:
            tag = n.plane.value.upper()
            c = n.content.strip()
            if len(c) > 300:
                c = c[:300] + "... [truncated]"
            ev_lines.append(f"• [{tag} - UNTRUSTED OBSERVATION] {c}")
        unprivileged_evidence = "\n".join(ev_lines)

        return privileged_invariants, unprivileged_evidence

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Deterministic SWE context retrieval with clearly demarcated privileged and unprivileged sections."""
        priv, unpriv = self.retrieve_two_slot(query, top_k=top_k)
        sections = []
        if priv:
            sections.append(f"=== PRIVILEGED ARCHITECTURAL INVARIANTS (PROVENANCE: SYSTEM/USER) ===\n{priv}")
        if unpriv:
            sections.append(f"=== UNPRIVILEGED EXECUTION CONTEXT (OBSERVED TOOL OUTPUT - UNTRUSTED) ===\n{unpriv}")
        return "\n\n".join(sections) if sections else "No memory records available."

    def get_stats(self) -> Dict:
        """Return memory engine telemetry and graph status."""
        with self._lock:
            planes_count = {p.value: 0 for p in MemoryPlane}
            pinned_count = 0
            for n in self.nodes_map.values():
                planes_count[n.plane.value] += 1
                if n.is_pinned:
                    pinned_count += 1

            return {
                "total_nodes": len(self.nodes_map),
                "total_edges": self.graph.number_of_edges(),
                "pinned_invariants": pinned_count,
                "plane_distribution": planes_count
            }

    def clear(self):
        """Reset the memory graph (used between test runs)."""
        with self._lock:
            self.graph.clear()
            self.nodes_map.clear()
            self.node_counter = 0
        logger.info("[LAYA-SWE] Memory graph cleared.")


# Backward compatibility aliases
AxiomMemoryEngine = LayaSweMemoryEngine
MultiRelationalMemoryEngine = LayaSweMemoryEngine


class SessionMemoryManager:
    """Multi-Agent Session Memory Manager providing strict isolation across agents/tasks."""

    def __init__(self, default_controller_mode: Optional[str] = None):
        self._default_mode = default_controller_mode or os.getenv("LAYA_CONTROLLER_MODE") or os.getenv("AXIOM_CONTROLLER_MODE", "laya_hybrid")
        self._sessions: Dict[str, Tuple[LayaSweMemoryEngine, Any]] = {}
        self._manager_lock = threading.Lock()

    def get_or_create(self, session_id: str, compactor_factory, controller_mode: Optional[str] = None) -> Tuple[LayaSweMemoryEngine, Any]:
        """Get or initialize isolated memory engine and compactor for a session."""
        with self._manager_lock:
            if session_id not in self._sessions:
                mode = controller_mode or self._default_mode
                controller = LayaMemoryController(mode=mode)
                engine = LayaSweMemoryEngine(controller=controller)
                compactor = compactor_factory(engine)
                self._sessions[session_id] = (engine, compactor)
            return self._sessions[session_id]

    def reset_session(self, session_id: Optional[str] = None):
        """Reset a specific session or all active sessions."""
        with self._manager_lock:
            if session_id:
                if session_id in self._sessions:
                    engine, _ = self._sessions[session_id]
                    engine.clear()
            else:
                for engine, _ in self._sessions.values():
                    engine.clear()
                self._sessions.clear()
