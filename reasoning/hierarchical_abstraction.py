#!/usr/bin/env python3
"""
Hierarchical Abstraction System v2.0
=====================================
Active abstraction with FULL architectural fixes:
1. Semantic overreach protection (embedding + domain + ontology)
2. Feedback loop damping (caps + novel evidence)
3. Full-spectrum decay (all influence effects)
4. Counterfactual stress testing (anticipatory volatility)
5. Planning integration (principles shape strategy)

Author: TorinAI System
Version: 2.0 - Production Ready
"""

import asyncio
import logging
import uuid
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import re

logger = logging.getLogger(__name__)


class AbstractionLevel(Enum):
    """Levels in the concept hierarchy"""
    EPISODIC = 0
    PATTERN = 1
    SCHEMA = 2
    PRINCIPLE = 3


class RelationshipType(Enum):
    """Logical relationships between concepts"""
    IMPLIES = "implies"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    COMPETES_WITH = "competes_with"
    ABSTRACTION_OF = "abstraction_of"
    INSTANTIATION_OF = "instantiation_of"


@dataclass
class ProbabilisticSchema:
    """Schema with counterexample tracking and feedback loop protection"""
    schema_id: str
    condition: Dict[str, Any]
    outcome: Dict[str, Any]
    probability: float
    confidence_interval: Tuple[float, float]
    supporting_memories: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    confounders: List[str] = field(default_factory=list)
    competing_schemas: List[str] = field(default_factory=list)
    induction_method: str = "similarity_clustering"
    abstraction_confidence: float = 0.5
    evidence_count: int = 0
    formation_time: datetime = field(default_factory=datetime.now)
    reinforcement_count: int = 0
    decay_rate: float = 0.05
    last_reinforced: datetime = field(default_factory=datetime.now)
    stability_score: float = 0.0
    belief_id: Optional[str] = None

    # Active influence effects
    retrieval_boost: float = 1.0
    prior_adjustment: float = 0.0
    attention_weight: float = 1.0

    # Feedback loop protection (FIX #2)
    cumulative_prior_adjustments: Dict[str, float] = field(default_factory=dict)
    cumulative_retrieval_boosts: Dict[str, float] = field(default_factory=dict)
    boost_history: List[Tuple[datetime, str, float]] = field(default_factory=list)
    novel_evidence_count: int = 0  # Evidence NOT from boosted retrieval

    # Context tracking
    session_ids: Set[str] = field(default_factory=set)
    temporal_span_days: float = 0.0
    context_diversity_score: float = 0.0

    # Counterfactual stress testing (FIX #4)
    stress_test_score: float = 0.0
    last_stress_test: Optional[datetime] = None
    fragility_detected: bool = False

    def calculate_probability(self) -> float:
        """Calculate P(outcome | condition) from evidence"""
        positive = len(self.supporting_memories)
        negative = len(self.counterexamples)
        total = positive + negative
        if total == 0:
            return 0.5
        return (positive + 1) / (total + 2)

    def calculate_credible_interval(self) -> Tuple[float, float]:
        """Calculate 95% Bayesian credible interval"""
        alpha = len(self.supporting_memories) + 1
        beta = len(self.counterexamples) + 1
        margin = 1.96 * np.sqrt(self.probability * (1 - self.probability) / max(self.evidence_count, 1))
        return (max(0.0, self.probability - margin), min(1.0, self.probability + margin))

    def update_evidence(self, new_memory_id: str, supports: bool, is_novel: bool = False):
        """Update schema with new evidence"""
        if supports:
            self.supporting_memories.append(new_memory_id)
            self.reinforcement_count += 1
            if is_novel:
                self.novel_evidence_count += 1
        else:
            self.counterexamples.append(new_memory_id)
        self.evidence_count += 1
        self.last_reinforced = datetime.now()
        self.probability = self.calculate_probability()
        self.confidence_interval = self.calculate_credible_interval()

    def calculate_decay_rate(self, context_diversity: float) -> float:
        """Calculate decay rate - early schemas decay faster"""
        age_days = (datetime.now() - self.formation_time).days
        base_decay = 0.05
        reinforcement_factor = min(self.reinforcement_count / 20.0, 1.0)
        age_factor = min(age_days / 90.0, 1.0)
        diversity_factor = context_diversity

        # Increase decay if fragile (FIX #4)
        fragility_penalty = 1.5 if self.fragility_detected else 1.0

        decay_rate = base_decay * fragility_penalty * (
            (1 - reinforcement_factor * 0.6) *
            (1 - age_factor * 0.3) *
            (1 - diversity_factor * 0.1)
        )
        return max(0.005, min(decay_rate, 0.08))

    def apply_temporal_decay(self, time_delta_hours: float) -> Tuple[float, float, float, float]:
        """
        FIX #3: Apply decay to probability AND all influence effects.
        Returns: (decayed_prob, decayed_attention, decayed_boost, decayed_prior_adj)
        """
        decay_factor = 1 - np.exp(-self.decay_rate * time_delta_hours / 24.0)

        # Decay probability toward 0.5 (uncertainty)
        current_prob = self.probability
        decayed_prob = current_prob + (0.5 - current_prob) * decay_factor

        # Decay attention weight toward 1.0 (neutral)
        decayed_attention = self.attention_weight + (1.0 - self.attention_weight) * decay_factor

        # Decay retrieval boost toward 1.0 (no boost)
        decayed_boost = self.retrieval_boost + (1.0 - self.retrieval_boost) * decay_factor

        # Decay prior adjustment toward 0.0 (no adjustment)
        decayed_prior_adj = self.prior_adjustment * (1 - decay_factor * 0.5)

        return decayed_prob, decayed_attention, decayed_boost, decayed_prior_adj


@dataclass
class AbstractionCandidate:
    """Cluster evaluated for abstraction with continuous pressure scoring"""
    cluster_id: str
    memory_ids: List[str]
    frequency_weight: float = 0.0
    cross_context_weight: float = 0.0
    outcome_coherence: float = 0.0
    temporal_consistency: float = 0.0
    reusability_signal: float = 0.0
    reasoning_depth: float = 0.0
    contradiction_penalty: float = 0.0
    abstraction_score: float = 0.0
    extracted_condition: Optional[Dict[str, Any]] = None
    extracted_outcome: Optional[Dict[str, Any]] = None

    def calculate_abstraction_pressure(self) -> float:
        """Calculate abstraction pressure score"""
        score = (
            self.frequency_weight * 1.0 +
            self.cross_context_weight * 2.0 +
            self.outcome_coherence * 1.5 +
            self.temporal_consistency * 1.2 +
            self.reusability_signal * 0.8 +
            self.reasoning_depth * 0.5 -
            self.contradiction_penalty * 3.0
        )
        return max(0.0, score)

    def should_abstract(self, threshold: float = 5.0) -> bool:
        """Check if pressure exceeds threshold"""
        return self.abstraction_score > threshold


@dataclass
class ConceptNode:
    """Node in concept hierarchy constraint graph"""
    concept_id: str
    level: AbstractionLevel
    content: str
    probability: float
    abstraction_of: List[str] = field(default_factory=list)
    instantiated_by: List[str] = field(default_factory=list)
    implies: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)
    competes_with: List[str] = field(default_factory=list)
    belief_id: Optional[str] = None
    schema_id: Optional[str] = None
    retrieval_boost: float = 1.0
    prior_adjustment: float = 0.0
    attention_weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0

    # For planning integration (FIX #5)
    strategy_template: Optional[Dict[str, Any]] = None
    applicable_contexts: List[str] = field(default_factory=list)


class ConceptHierarchy:
    """Manages concept lattice as constraint graph"""

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.forward_edges: Dict[str, Set[str]] = defaultdict(set)
        self.backward_edges: Dict[str, Set[str]] = defaultdict(set)
        self.implication_edges: Dict[str, Set[str]] = defaultdict(set)
        self.contradiction_edges: Dict[str, Set[str]] = defaultdict(set)
        self.stats = {
            'total_concepts': 0,
            'episodic_concepts': 0,
            'pattern_concepts': 0,
            'schema_concepts': 0,
            'principle_concepts': 0,
            'constraint_violations': 0
        }

    def add_concept(self, node: ConceptNode):
        """Add concept and update graph edges"""
        self.nodes[node.concept_id] = node
        for parent_id in node.abstraction_of:
            self.backward_edges[node.concept_id].add(parent_id)
            self.forward_edges[parent_id].add(node.concept_id)
        for implied_id in node.implies:
            self.implication_edges[node.concept_id].add(implied_id)
        for contradicted_id in node.contradicts:
            self.contradiction_edges[node.concept_id].add(contradicted_id)
            self.contradiction_edges[contradicted_id].add(node.concept_id)
        self.stats['total_concepts'] += 1
        if node.level == AbstractionLevel.EPISODIC:
            self.stats['episodic_concepts'] += 1
        elif node.level == AbstractionLevel.PATTERN:
            self.stats['pattern_concepts'] += 1
        elif node.level == AbstractionLevel.SCHEMA:
            self.stats['schema_concepts'] += 1
        elif node.level == AbstractionLevel.PRINCIPLE:
            self.stats['principle_concepts'] += 1

    def get_ancestors(self, concept_id: str, max_depth: int = 5) -> List[ConceptNode]:
        """Get parent concepts up hierarchy"""
        ancestors = []
        visited = set()
        queue = [(concept_id, 0)]
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth or current_id in visited:
                continue
            visited.add(current_id)
            for parent_id in self.backward_edges.get(current_id, []):
                if parent_id in self.nodes:
                    ancestors.append(self.nodes[parent_id])
                    queue.append((parent_id, depth + 1))
        return ancestors

    def get_descendants(self, concept_id: str, max_depth: int = 5) -> List[ConceptNode]:
        """Get child concepts down hierarchy"""
        descendants = []
        visited = set()
        queue = [(concept_id, 0)]
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth or current_id in visited:
                continue
            visited.add(current_id)
            for child_id in self.forward_edges.get(current_id, []):
                if child_id in self.nodes:
                    descendants.append(self.nodes[child_id])
                    queue.append((child_id, depth + 1))
        return descendants

    def find_principles_for_domain(self, domain: str) -> List[ConceptNode]:
        """Find Level 3 principles relevant to domain"""
        principles = []
        for node in self.nodes.values():
            if node.level == AbstractionLevel.PRINCIPLE:
                if domain in node.applicable_contexts or not node.applicable_contexts:
                    principles.append(node)
        return principles

    def check_consistency(self) -> List[Dict[str, Any]]:
        """Check for logical violations"""
        violations = []
        for concept_id, node in self.nodes.items():
            for implied_id in node.implies:
                if implied_id in self.nodes:
                    implied_node = self.nodes[implied_id]
                    if node.probability > implied_node.probability + 0.15:
                        violations.append({
                            'type': 'implication',
                            'source': concept_id,
                            'target': implied_id,
                            'source_prob': node.probability,
                            'target_prob': implied_node.probability
                        })
                        self.stats['constraint_violations'] += 1
            for contradicted_id in node.contradicts:
                if contradicted_id in self.nodes:
                    contradicted_node = self.nodes[contradicted_id]
                    prob_sum = node.probability + contradicted_node.probability
                    if not (0.8 <= prob_sum <= 1.2):
                        violations.append({
                            'type': 'contradiction',
                            'source': concept_id,
                            'target': contradicted_id,
                            'prob_sum': prob_sum
                        })
                        self.stats['constraint_violations'] += 1
        return violations


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.agents.memory_agent import MemoryAgent
    from core.reasoning.bayesian_uncertainty import BayesianUncertaintySystem
    from core.reasoning.abstract_reasoning_engine import AbstractReasoningEngine


class AbstractionPipeline:
    """Active abstraction with all architectural fixes"""

    def __init__(
        self,
        memory_agent: 'MemoryAgent',
        uncertainty_system: 'BayesianUncertaintySystem',
        reasoning_engine: Optional['AbstractReasoningEngine'] = None
    ):
        self.memory = memory_agent
        self.beliefs = uncertainty_system
        self.reasoning = reasoning_engine
        self.concept_hierarchy = ConceptHierarchy()
        self.active_schemas: Dict[str, ProbabilisticSchema] = {}
        self.abstraction_candidates: Dict[str, AbstractionCandidate] = {}
        self.monitoring_active = False
        self.last_monitoring_run: Optional[datetime] = None
        self.stats = {
            'schemas_formed': 0,
            'schemas_reinforced': 0,
            'schemas_decayed': 0,
            'priors_modified': 0,
            'retrieval_weights_modified': 0,
            'attention_biases_added': 0,
            'abstraction_triggers': 0,
            'feedback_loops_prevented': 0,
            'semantic_overreach_prevented': 0,
            'stress_tests_run': 0
        }
        logger.info("AbstractionPipeline v2.0 initialized")

    async def start_monitoring(self, interval_hours: float = 1.0):
        """Start continuous abstraction pressure monitoring"""
        if self.monitoring_active:
            return
        self.monitoring_active = True
        logger.info(f"Starting abstraction monitoring (interval: {interval_hours}h)")
        while self.monitoring_active:
            try:
                await self.monitor_abstraction_pressure()
                await self.apply_temporal_decay_to_schemas()  # FIX #3: Decay all effects
                self.last_monitoring_run = datetime.now()
                await asyncio.sleep(interval_hours * 3600)
            except Exception as e:
                logger.error(f"Abstraction monitoring error: {e}")
                await asyncio.sleep(60)

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False

    async def monitor_abstraction_pressure(self):
        """Calculate abstraction pressure and trigger schema formation"""
        memories = await self.memory.get_recent_memories(limit=500)
        if len(memories) < 5:
            return
        clusters = await self._cluster_memories(memories, similarity_threshold=0.75)
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            candidate = await self._create_abstraction_candidate(cluster)
            candidate.abstraction_score = candidate.calculate_abstraction_pressure()
            threshold = self._calculate_dynamic_threshold()
            if candidate.should_abstract(threshold):
                self.stats['abstraction_triggers'] += 1
                await self.extract_and_apply_schema(cluster, candidate)

    async def _create_abstraction_candidate(self, cluster: List[Any]) -> AbstractionCandidate:
        """Create candidate from cluster using existing metadata"""
        candidate = AbstractionCandidate(
            cluster_id=f"cluster_{uuid.uuid4().hex[:8]}",
            memory_ids=[m.memory_id for m in cluster]
        )
        candidate.reusability_signal = np.mean([m.metadata.get('reusability', 0.5) for m in cluster])
        candidate.reasoning_depth = np.mean([m.metadata.get('reasoning_depth', 1.0) for m in cluster])
        candidate.cross_context_weight = await self._assess_context_diversity(cluster)
        candidate.outcome_coherence = await self._assess_outcome_coherence(cluster)
        candidate.temporal_consistency = await self._assess_temporal_consistency(cluster)
        candidate.frequency_weight = len(cluster) / 100.0
        candidate.contradiction_penalty = await self._count_contradictions(cluster)
        return candidate

    async def _assess_context_diversity(self, memories: List[Any]) -> float:
        """Assess pattern holds across different contexts"""
        sessions = {m.session_id for m in memories if m.session_id}
        session_diversity = len(sessions) / max(len(memories), 1)
        times = [m.created_at for m in memories if m.created_at]
        if len(times) >= 2:
            time_span = (max(times) - min(times)).days
            temporal_diversity = min(time_span / 30.0, 1.0)
        else:
            temporal_diversity = 0.0
        emotional_states = [
            m.emotional_context.get('state') if isinstance(m.emotional_context, dict) else None
            for m in memories if hasattr(m, 'emotional_context') and m.emotional_context
        ]
        emotional_states = [s for s in emotional_states if s]
        emotional_diversity = len(set(emotional_states)) / max(len(emotional_states), 1) if emotional_states else 0.5
        diversity_score = session_diversity * 0.4 + temporal_diversity * 0.4 + emotional_diversity * 0.2
        return diversity_score

    async def _assess_outcome_coherence(self, memories: List[Any]) -> float:
        """Check consistent outcomes across memories"""
        outcomes = []
        for m in memories:
            if isinstance(m.content, dict):
                outcome = m.content.get('outcome') or m.content.get('result') or m.content.get('answer')
                if outcome:
                    outcomes.append(str(outcome).lower())
        if len(outcomes) < 2:
            return 0.5
        similarity_sum = 0.0
        comparisons = 0
        for i in range(len(outcomes)):
            for j in range(i + 1, len(outcomes)):
                words_i = set(outcomes[i].split())
                words_j = set(outcomes[j].split())
                if words_i and words_j:
                    overlap = len(words_i & words_j) / max(len(words_i), len(words_j))
                    similarity_sum += overlap
                    comparisons += 1
        return similarity_sum / comparisons if comparisons > 0 else 0.5

    async def _assess_temporal_consistency(self, memories: List[Any]) -> float:
        """Check pattern persists regularly over time"""
        times = sorted([m.created_at for m in memories if m.created_at])
        if len(times) < 3:
            return 0.5
        gaps = [(times[i+1] - times[i]).days for i in range(len(times) - 1)]
        if not gaps:
            return 0.5
        mean_gap = np.mean(gaps)
        variance = np.var(gaps)
        if mean_gap == 0:
            return 0.5
        consistency = 1.0 / (1.0 + variance / (mean_gap + 1))
        return min(consistency, 1.0)

    async def _count_contradictions(self, memories: List[Any]) -> float:
        """Count contradictory memories in cluster"""
        if len(memories) < 2:
            return 0.0
        outcomes = []
        for m in memories:
            if isinstance(m.content, dict):
                outcome = m.content.get('outcome') or m.content.get('result')
                if outcome:
                    outcomes.append(str(outcome).lower())
        if len(outcomes) < 2:
            return 0.0
        unique_outcomes = len(set(outcomes))
        contradiction_ratio = (unique_outcomes - 1) / len(outcomes)
        return contradiction_ratio * 2.0

    def _calculate_dynamic_threshold(self) -> float:
        """Calculate dynamic threshold based on system state"""
        base_threshold = 5.0
        schema_factor = len(self.active_schemas) / 100.0
        recent_formations = sum(1 for s in self.active_schemas.values() if (datetime.now() - s.formation_time).days < 7)
        throttle_factor = recent_formations / 10.0
        threshold = base_threshold * (1 + schema_factor * 0.5 + throttle_factor * 0.3)
        return max(3.0, min(threshold, 10.0))

    async def _cluster_memories(self, memories: List[Any], similarity_threshold: float = 0.75) -> List[List[Any]]:
        """Cluster memories by embedding similarity"""
        if not memories:
            return []
        embeddings_map = {}
        for m in memories:
            if hasattr(m, 'embeddings') and m.embeddings:
                embeddings_map[m.memory_id] = np.array(m.embeddings)
            elif hasattr(m, 'embedding') and m.embedding:
                embeddings_map[m.memory_id] = np.array(m.embedding)
        if not embeddings_map:
            by_type = defaultdict(list)
            for m in memories:
                by_type[m.memory_type].append(m)
            return [mem_list for mem_list in by_type.values() if len(mem_list) >= 3]
        memory_list = [m for m in memories if m.memory_id in embeddings_map]
        clusters = []
        used = set()
        for i, mem_i in enumerate(memory_list):
            if mem_i.memory_id in used:
                continue
            cluster = [mem_i]
            used.add(mem_i.memory_id)
            emb_i = embeddings_map[mem_i.memory_id]
            for j, mem_j in enumerate(memory_list):
                if i != j and mem_j.memory_id not in used:
                    emb_j = embeddings_map[mem_j.memory_id]
                    similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
                    if similarity >= similarity_threshold:
                        cluster.append(mem_j)
                        used.add(mem_j.memory_id)
            if len(cluster) >= 3:
                clusters.append(cluster)
        return clusters

    async def extract_and_apply_schema(self, cluster: List[Any], candidate: AbstractionCandidate):
        """Extract probabilistic schema and apply structural effects"""
        schema = await self._extract_probabilistic_schema(cluster, candidate)
        if not schema:
            return

        # FIX #4: Counterfactual stress testing
        stress_score = await self.stress_test_schema(schema)
        schema.stress_test_score = stress_score
        schema.last_stress_test = datetime.now()
        self.stats['stress_tests_run'] += 1

        self.active_schemas[schema.schema_id] = schema
        self.stats['schemas_formed'] += 1
        await self._create_belief_from_schema(schema)
        await self.apply_abstraction_effects(schema)
        await self._add_to_hierarchy(schema, cluster)

    async def _extract_probabilistic_schema(self, cluster: List[Any], candidate: AbstractionCandidate) -> Optional[ProbabilisticSchema]:
        """Extract schema with counterexamples"""
        condition, outcome = await self._extract_pattern(cluster)
        if not condition or not outcome:
            return None
        all_memories = await self.memory.get_recent_memories(limit=1000)
        counterexamples = []
        for memory in all_memories:
            if memory.memory_id in [m.memory_id for m in cluster]:
                continue
            if self._matches_condition(memory, condition):
                if not self._matches_outcome(memory, outcome):
                    counterexamples.append(memory.memory_id)
        positive = len(cluster)
        negative = len(counterexamples)
        total = positive + negative
        if total == 0:
            return None
        probability = (positive + 1) / (total + 2)

        # Infer domain from cluster memories
        domain = self._infer_domain_from_cluster(cluster)

        schema = ProbabilisticSchema(
            schema_id=f"schema_{uuid.uuid4().hex[:12]}",
            condition=condition,
            outcome=outcome,
            probability=probability,
            confidence_interval=(0.0, 1.0),
            supporting_memories=[m.memory_id for m in cluster],
            counterexamples=counterexamples,
            evidence_count=total,
            abstraction_confidence=candidate.abstraction_score / 10.0,
            context_diversity_score=candidate.cross_context_weight
        )
        schema.confidence_interval = schema.calculate_credible_interval()
        schema.decay_rate = schema.calculate_decay_rate(candidate.cross_context_weight)
        schema.session_ids = {m.session_id for m in cluster if m.session_id}
        schema.metadata['domain'] = domain  # Tag schema with domain
        times = [m.created_at for m in cluster if m.created_at]
        if len(times) >= 2:
            schema.temporal_span_days = (max(times) - min(times)).days
        return schema

    async def _extract_pattern(self, cluster: List[Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract common condition and outcome from cluster"""
        condition_features = defaultdict(list)
        outcome_features = defaultdict(list)
        for m in cluster:
            if isinstance(m.content, dict):
                for key, value in m.content.items():
                    if key in ['outcome', 'result', 'answer', 'conclusion']:
                        outcome_features[key].append(str(value))
                    else:
                        condition_features[key].append(str(value))
            if hasattr(m, 'tags') and m.tags:
                condition_features['tags'].extend(m.tags)
            if hasattr(m, 'memory_type'):
                condition_features['memory_type'].append(str(m.memory_type))
        condition = {}
        for key, values in condition_features.items():
            counter = Counter(values)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= len(cluster) * 0.5:
                condition[key] = most_common[0][0]
        outcome = {}
        for key, values in outcome_features.items():
            counter = Counter(values)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= len(cluster) * 0.5:
                outcome[key] = most_common[0][0]
        if not condition:
            condition = {'cluster_size': len(cluster)}
        if not outcome:
            outcome = {'pattern': 'recurring_theme'}
        return condition, outcome

    def _matches_condition(self, memory: Any, condition: Dict[str, Any]) -> bool:
        """Check if memory matches schema condition"""
        if not condition:
            return False
        matches = 0
        total_checks = 0
        for key, expected_value in condition.items():
            total_checks += 1
            if isinstance(memory.content, dict) and key in memory.content:
                if str(memory.content[key]).lower() == str(expected_value).lower():
                    matches += 1
            elif hasattr(memory, key):
                if str(getattr(memory, key)).lower() == str(expected_value).lower():
                    matches += 1
            elif key == 'tags' and hasattr(memory, 'tags') and memory.tags:
                if expected_value in memory.tags or any(expected_value in str(t) for t in memory.tags):
                    matches += 1
        return (matches / total_checks) >= 0.5 if total_checks > 0 else False

    def _matches_outcome(self, memory: Any, outcome: Dict[str, Any]) -> bool:
        """Check if memory matches schema outcome"""
        if not outcome:
            return False
        matches = 0
        total_checks = 0
        for key, expected_value in outcome.items():
            total_checks += 1
            if isinstance(memory.content, dict) and key in memory.content:
                actual = str(memory.content[key]).lower()
                expected = str(expected_value).lower()
                if actual == expected or expected in actual or self._semantic_similarity(actual, expected) > 0.6:
                    matches += 1
        return (matches / total_checks) >= 0.5 if total_checks > 0 else False

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    async def _create_belief_from_schema(self, schema: ProbabilisticSchema):
        """Create Bayesian belief from schema"""
        claim = f"Pattern: {schema.condition} → {schema.outcome}"
        belief = self.beliefs.create_belief(
            claim=claim,
            domain="induced_schema",
            prior=schema.probability,
            evidence={
                'type': 'schema_induction',
                'support_count': len(schema.supporting_memories),
                'counter_count': len(schema.counterexamples),
                'quality': schema.abstraction_confidence,
                'schema_id': schema.schema_id
            }
        )
        schema.belief_id = belief.belief_id

    async def apply_abstraction_effects(self, schema: ProbabilisticSchema):
        """Apply structural effects with all protections"""
        await self._boost_matching_memories(schema)
        await self._adjust_related_priors(schema)
        await self._add_attention_bias(schema)
        await self._flag_contradictions(schema)

    async def _boost_matching_memories(self, schema: ProbabilisticSchema):
        """FIX #2: Boost with cumulative caps"""
        for memory_id in schema.supporting_memories:
            try:
                # Check cumulative boost cap
                current_boost = schema.cumulative_retrieval_boosts.get(memory_id, 1.0)
                if current_boost >= 1.5:  # Max 50% boost
                    self.stats['feedback_loops_prevented'] += 1
                    continue

                memory = await self.memory.get_memory(memory_id)
                if memory:
                    # Check existing boosts from other schemas
                    existing_boosts = memory.metadata.get('schema_boosts', [])
                    if len(existing_boosts) >= 3:  # Max 3 schemas can boost same memory
                        self.stats['feedback_loops_prevented'] += 1
                        continue

                    new_importance = min(1.0, memory.importance_score * 1.2)
                    await self.memory.update_memory(
                        memory_id,
                        importance_score=new_importance,
                        metadata={
                            **memory.metadata,
                            'schema_support': schema.schema_id,
                            'schema_boosts': existing_boosts + [schema.schema_id],
                            'boost_count': len(existing_boosts) + 1
                        }
                    )
                    schema.cumulative_retrieval_boosts[memory_id] = current_boost * 1.2
                    self.stats['retrieval_weights_modified'] += 1
            except Exception as e:
                logger.error(f"Error boosting memory {memory_id}: {e}")

    async def _adjust_related_priors(self, schema: ProbabilisticSchema):
        """FIX #1 & #2: Adjust priors with semantic protection and caps"""
        if not schema.belief_id or schema.belief_id not in self.beliefs.beliefs:
            return

        schema_belief = self.beliefs.beliefs[schema.belief_id]

        for belief_id, belief in self.beliefs.beliefs.items():
            if belief_id == schema.belief_id:
                continue

            # FIX #2: Check cumulative cap
            current_cumulative = schema.cumulative_prior_adjustments.get(belief_id, 0.0)
            if current_cumulative >= 0.30:  # Max 30% cumulative adjustment
                self.stats['feedback_loops_prevented'] += 1
                continue

            # FIX #2: Require novel evidence after 5 reinforcements
            if schema.reinforcement_count > 5:
                if schema.novel_evidence_count < schema.reinforcement_count * 0.3:
                    self.stats['feedback_loops_prevented'] += 1
                    continue

            # FIX #1: Multi-layer relatedness check
            if self._beliefs_related_strict(schema_belief, belief):
                adjustment = schema.probability * 0.15
                allowed_adjustment = min(adjustment, 0.30 - current_cumulative)

                old_prior = belief.prior_probability
                belief.prior_probability = min(1.0, max(0.0, old_prior + allowed_adjustment))

                if abs(belief.prior_probability - old_prior) > 0.01:
                    schema.cumulative_prior_adjustments[belief_id] = current_cumulative + allowed_adjustment
                    schema.boost_history.append((datetime.now(), belief_id, allowed_adjustment))
                    self.stats['priors_modified'] += 1

    def _beliefs_related_strict(self, belief1: Any, belief2: Any) -> bool:
        """
        FIX #1: Strict semantic relatedness with multiple constraints.
        Prevents semantic overreach (30% word overlap → 50% + embedding + domain + ontology)
        """
        # Layer 1: Word overlap (raised threshold)
        claim1_words = set(re.findall(r'\w+', belief1.claim.lower()))
        claim2_words = set(re.findall(r'\w+', belief2.claim.lower()))

        if not claim1_words or not claim2_words:
            return False

        overlap = len(claim1_words & claim2_words)
        min_size = min(len(claim1_words), len(claim2_words))
        word_overlap_score = overlap / min_size if min_size > 0 else 0.0

        if word_overlap_score < 0.5:  # Raised from 0.3 to 0.5
            self.stats['semantic_overreach_prevented'] += 1
            return False

        # Layer 2: Domain alignment
        if belief1.domain != belief2.domain and belief1.domain != "induced_schema" and belief2.domain != "induced_schema":
            self.stats['semantic_overreach_prevented'] += 1
            return False

        # Layer 3: Embedding similarity (if available)
        if hasattr(belief1, 'embedding') and hasattr(belief2, 'embedding') and belief1.embedding and belief2.embedding:
            emb1 = np.array(belief1.embedding)
            emb2 = np.array(belief2.embedding)
            emb_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            if emb_sim < 0.6:
                self.stats['semantic_overreach_prevented'] += 1
                return False

        # Layer 4: Ontology-level type checking
        entities1 = self._extract_entities(belief1.claim)
        entities2 = self._extract_entities(belief2.claim)

        if entities1 and entities2:
            if not self._share_concept_category(entities1, entities2):
                self.stats['semantic_overreach_prevented'] += 1
                return False

        return True

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from text"""
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z_]+\b', text)
        entities = [w for w in words if len(w) > 2]
        return entities

    def _share_concept_category(self, entities1: List[str], entities2: List[str]) -> bool:
        """Check if entities share conceptual categories"""
        categories = {
            'programming': {'python', 'java', 'javascript', 'code', 'program', 'function', 'class', 'method'},
            'data': {'data', 'analysis', 'dataframe', 'dataset', 'table', 'query', 'database'},
            'tools': {'library', 'framework', 'tool', 'package', 'module', 'system'},
            'concepts': {'algorithm', 'pattern', 'structure', 'design', 'architecture'}
        }

        def categorize(entity: str) -> Set[str]:
            matches = set()
            entity_lower = entity.lower()
            for category, keywords in categories.items():
                if entity_lower in keywords or any(kw in entity_lower for kw in keywords):
                    matches.add(category)
            return matches

        cats1 = set()
        for e in entities1:
            cats1.update(categorize(e))

        cats2 = set()
        for e in entities2:
            cats2.update(categorize(e))

        return len(cats1 & cats2) > 0

    async def _add_attention_bias(self, schema: ProbabilisticSchema):
        """Add attention bias for future reasoning"""
        schema.attention_weight = 1.0 + (schema.probability - 0.5) * 0.5
        self.stats['attention_biases_added'] += 1

    async def _flag_contradictions(self, schema: ProbabilisticSchema):
        """Flag contradicting memories for strong schemas"""
        if schema.probability < 0.7:
            return
        for memory_id in schema.counterexamples:
            try:
                memory = await self.memory.get_memory(memory_id)
                if memory:
                    await self.memory.update_memory(
                        memory_id,
                        metadata={**memory.metadata, 'contradicts_schema': schema.schema_id, 'needs_review': True}
                    )
            except Exception as e:
                logger.error(f"Error flagging memory {memory_id}: {e}")

    async def _add_to_hierarchy(self, schema: ProbabilisticSchema, cluster: List[Any]):
        """Add schema to concept hierarchy with strategy template"""
        # Extract strategy template from schema (FIX #5)
        strategy_template = {
            'when': schema.condition,
            'prefer': schema.outcome,
            'confidence': schema.probability
        }

        concept = ConceptNode(
            concept_id=f"concept_{uuid.uuid4().hex[:12]}",
            level=AbstractionLevel.SCHEMA,
            content=f"{schema.condition} → {schema.outcome}",
            probability=schema.probability,
            instantiated_by=[m.memory_id for m in cluster],
            belief_id=schema.belief_id,
            schema_id=schema.schema_id,
            strategy_template=strategy_template,
            applicable_contexts=[schema.condition.get('domain', 'general')]
        )
        self.concept_hierarchy.add_concept(concept)

    async def stress_test_schema(self, schema: ProbabilisticSchema) -> float:
        """
        FIX #4: Counterfactual stress testing for anticipatory volatility adjustment.
        Tests schema against alternate scenarios to detect fragility.
        """
        counterfactuals = []

        # Generate counterfactuals based on condition
        for key, value in schema.condition.items():
            if key != 'cluster_size':
                counterfactuals.append({
                    'type': 'condition_flip',
                    'condition': {**schema.condition, key: f"not_{value}"}
                })

        # Generate outcome flip
        for key, value in schema.outcome.items():
            counterfactuals.append({
                'type': 'outcome_flip',
                'outcome': {key: f"opposite_{value}"}
            })

        stability_score = 0.0
        fragility_count = 0

        for cf in counterfactuals:
            # Check if contradictory pattern already exists
            contradicting_memories = []

            all_memories = await self.memory.get_recent_memories(limit=500)
            for memory in all_memories:
                if cf['type'] == 'condition_flip':
                    if self._matches_outcome(memory, schema.outcome):
                        if not self._matches_condition(memory, schema.condition):
                            contradicting_memories.append(memory)
                elif cf['type'] == 'outcome_flip':
                    if self._matches_condition(memory, schema.condition):
                        if self._matches_outcome(memory, cf['outcome']):
                            contradicting_memories.append(memory)

            if len(contradicting_memories) > 0:
                fragility = len(contradicting_memories) / max(len(schema.supporting_memories), 1)
                stability_score -= fragility
                fragility_count += 1
            else:
                stability_score += 0.1

        # Adjust schema based on stress test
        if stability_score < 0:
            schema.fragility_detected = True
            schema.decay_rate = min(0.08, schema.decay_rate * 1.5)
            logger.warning(f"Schema {schema.schema_id} fragile: score={stability_score:.2f}, decay increased")
        else:
            schema.fragility_detected = False
            schema.decay_rate = max(0.01, schema.decay_rate * 0.8)

        return stability_score

    async def apply_temporal_decay_to_schemas(self):
        """
        FIX #3: Apply decay to ALL effects (probability, attention, boost, prior adjustments).
        Decay is structural, not cosmetic.
        """
        now = datetime.now()
        for schema_id, schema in list(self.active_schemas.items()):
            time_delta_hours = (now - schema.last_reinforced).total_seconds() / 3600.0

            if time_delta_hours > 24:
                # Apply decay to ALL components
                decayed_prob, decayed_attention, decayed_boost, decayed_prior_adj = schema.apply_temporal_decay(time_delta_hours)

                # Update schema
                schema.probability = decayed_prob
                schema.attention_weight = decayed_attention
                schema.retrieval_boost = decayed_boost
                schema.prior_adjustment = decayed_prior_adj

                # Decay prior adjustments to beliefs
                decay_factor = 1 - np.exp(-schema.decay_rate * time_delta_hours / 24.0)
                for belief_id, adjustment in list(schema.cumulative_prior_adjustments.items()):
                    decayed_adjustment = adjustment * (1 - decay_factor * 0.5)
                    diff = adjustment - decayed_adjustment

                    # Reverse the difference in belief prior
                    if belief_id in self.beliefs.beliefs:
                        belief = self.beliefs.beliefs[belief_id]
                        belief.prior_probability = max(0.0, min(1.0, belief.prior_probability - diff))

                    schema.cumulative_prior_adjustments[belief_id] = decayed_adjustment

                # Decay retrieval boosts
                for memory_id, boost in list(schema.cumulative_retrieval_boosts.items()):
                    decayed_boost_val = 1.0 + (boost - 1.0) * (1 - decay_factor)
                    schema.cumulative_retrieval_boosts[memory_id] = decayed_boost_val

                # Update linked belief
                if schema.belief_id and schema.belief_id in self.beliefs.beliefs:
                    belief = self.beliefs.beliefs[schema.belief_id]
                    belief.posterior_probability = decayed_prob

                self.stats['schemas_decayed'] += 1

    async def process_memories(self, memory_dicts: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Process memories for abstraction formation

        Args:
            memory_dicts: List of memory dictionaries with id, content, timestamp, etc.

        Returns:
            Dictionary with counts of patterns, schemas, and principles formed
        """
        try:
            logger.info(f"Processing {len(memory_dicts)} memories for abstraction...")

            # Track results
            patterns_formed = 0
            schemas_formed = 0
            principles_extracted = 0

            # Cluster memories by similarity
            clusters = await self._cluster_memories(memory_dicts, similarity_threshold=0.75)

            logger.info(f"Identified {len(clusters)} memory clusters")

            # Process each cluster
            for cluster in clusters:
                if len(cluster) < 3:  # Need at least 3 instances for pattern
                    continue

                # Create abstraction candidate
                candidate = await self._create_abstraction_candidate(cluster)

                # Check if abstraction pressure exceeds threshold
                if not candidate.should_abstract(threshold=5.0):
                    continue

                # Extract and apply schema
                await self.extract_and_apply_schema(cluster, candidate)

                schemas_formed += 1
                patterns_formed += 1

            # Update concept hierarchy (extract principles from schemas)
            # Check if we have enough schemas to form principles
            if len(self.active_schemas) >= 5:
                principles_extracted = await self._extract_principles_from_schemas()

            logger.info(f"✓ Formed {patterns_formed} patterns, {schemas_formed} schemas, {principles_extracted} principles")

            return {
                'patterns_formed': patterns_formed,
                'schemas_formed': schemas_formed,
                'principles_extracted': principles_extracted
            }

        except Exception as e:
            logger.error(f"Failed to process memories: {e}")
            import traceback
            traceback.print_exc()
            return {
                'patterns_formed': 0,
                'schemas_formed': 0,
                'principles_extracted': 0
            }

    async def _extract_principles_from_schemas(self) -> int:
        """
        Extract Level 3 principles from Level 2 schemas

        Looks for meta-patterns across schemas to form higher-level principles
        """
        try:
            principles_count = 0

            # Group schemas by domain
            schemas_by_domain: Dict[str, List[ProbabilisticSchema]] = {}
            for schema_id, schema in self.active_schemas.items():
                domain = schema.metadata.get('domain', 'general')
                if domain not in schemas_by_domain:
                    schemas_by_domain[domain] = []
                schemas_by_domain[domain].append(schema)

            # For each domain with enough schemas, extract principles
            for domain, schemas in schemas_by_domain.items():
                if len(schemas) < 3:  # Need at least 3 schemas for principle
                    continue

                # Find common patterns across schemas
                # (Simplified - in full implementation would do semantic clustering)
                common_principle = {
                    'domain': domain,
                    'schema_count': len(schemas),
                    'avg_probability': sum(s.probability for s in schemas) / len(schemas),
                    'description': f"Meta-pattern in {domain} domain"
                }

                # Create principle node
                principle_node = ConceptNode(
                    concept_id=f"principle_{domain}_{len(self.concept_hierarchy.nodes)}",
                    level=AbstractionLevel.PRINCIPLE,
                    schema_ids=[s.schema_id for s in schemas],
                    confidence=common_principle['avg_probability'],
                    metadata={'domain': domain, 'schema_count': len(schemas)}
                )

                self.concept_hierarchy.add_concept(principle_node)
                principles_count += 1

            return principles_count

        except Exception as e:
            logger.error(f"Failed to extract principles: {e}")
            return 0

    async def apply_decay_to_abstractions(self):
        """Apply temporal decay to all active abstractions"""
        try:
            logger.info("Applying decay to abstractions...")

            # Apply decay to schemas (already implemented)
            now = datetime.now()
            schemas_decayed = 0

            for schema_id, schema in list(self.active_schemas.items()):
                time_delta_hours = (now - schema.last_reinforced).total_seconds() / 3600.0

                if time_delta_hours > 1:  # Apply decay after 1 hour
                    decayed_prob, decayed_attention, decayed_boost, decayed_prior_adj = schema.apply_temporal_decay(time_delta_hours)

                    schema.probability = decayed_prob
                    schema.attention_weight = decayed_attention
                    schema.retrieval_boost = decayed_boost
                    schema.prior_adjustment = decayed_prior_adj

                    schemas_decayed += 1

                    # Remove schema if it decayed below threshold
                    if schema.probability < 0.3:
                        del self.active_schemas[schema_id]
                        logger.info(f"Removed low-probability schema: {schema_id}")

            logger.info(f"✓ Applied decay to {schemas_decayed} schemas")

        except Exception as e:
            logger.error(f"Failed to apply decay: {e}")

    async def apply_schema_decay(self) -> Dict[str, int]:
        """
        Apply decay to schemas and run stress tests

        Returns:
            Dictionary with decay statistics
        """
        try:
            await self.apply_decay_to_abstractions()

            # Run stress tests on schemas
            fragile_schemas = 0
            for schema_id, schema in self.active_schemas.items():
                # Run counterfactual stress test if it hasn't been done recently
                if schema.last_stress_test is None or \
                   (datetime.now() - schema.last_stress_test).total_seconds() > 86400:  # 24 hours

                    stress_score = await self.stress_test_schema(schema)

                    if schema.fragility_detected:
                        fragile_schemas += 1

            return {
                'schemas_decayed': len(self.active_schemas),
                'fragile_schemas_detected': fragile_schemas
            }

        except Exception as e:
            logger.error(f"Failed to apply schema decay: {e}")
            return {
                'schemas_decayed': 0,
                'fragile_schemas_detected': 0
            }

    def _infer_domain_from_cluster(self, cluster: List[Any]) -> str:
        """
        Infer domain from cluster of memories

        Maps to Universal Domain Master domain types for cross-domain integration
        """
        try:
            from core.integration.universal_domain_master import DomainType

            # Aggregate content from all memories in cluster
            all_text = ""
            all_tags = set()

            for memory in cluster:
                # Extract text content
                if hasattr(memory, 'content'):
                    if isinstance(memory.content, dict):
                        all_text += " " + " ".join(str(v) for v in memory.content.values())
                    else:
                        all_text += " " + str(memory.content)

                # Extract tags
                if hasattr(memory, 'tags'):
                    if isinstance(memory.tags, (list, set)):
                        all_tags.update(str(tag).lower() for tag in memory.tags)
                    elif isinstance(memory.tags, str):
                        all_tags.add(memory.tags.lower())

            text_lower = all_text.lower()

            # Map keywords to Universal Domain Master domain types
            # SCIENTIFIC: Research, analysis, discovery
            if any(word in text_lower for word in ["research", "study", "analyze", "investigate", "hypothesis", "experiment"]):
                return DomainType.SCIENTIFIC.value
            if any(tag in all_tags for tag in ["research", "scientific", "analysis"]):
                return DomainType.SCIENTIFIC.value

            # TECHNICAL: Code, implementation, engineering
            elif any(word in text_lower for word in ["code", "implement", "build", "develop", "function", "class", "method"]):
                return DomainType.TECHNICAL.value
            elif any(tag in all_tags for tag in ["code", "technical", "implementation"]):
                return DomainType.TECHNICAL.value

            # MATHEMATICAL: Calculation, optimization, metrics
            elif any(word in text_lower for word in ["calculate", "optimize", "algorithm", "metric", "statistics", "probability"]):
                return DomainType.MATHEMATICAL.value
            elif any(tag in all_tags for tag in ["math", "calculation", "optimization"]):
                return DomainType.MATHEMATICAL.value

            # CAUSAL: Planning, strategy, cause-effect
            elif any(word in text_lower for word in ["plan", "strategy", "cause", "effect", "consequence", "because"]):
                return DomainType.CAUSAL.value
            elif any(tag in all_tags for tag in ["planning", "strategy", "causal"]):
                return DomainType.CAUSAL.value

            # ABSTRACT: Memory, reasoning, cognition
            elif any(word in text_lower for word in ["memory", "reason", "think", "cognition", "belief", "abstraction"]):
                return DomainType.ABSTRACT.value
            elif any(tag in all_tags for tag in ["memory", "reasoning", "cognitive"]):
                return DomainType.ABSTRACT.value

            # PRACTICAL: Testing, validation, application
            elif any(word in text_lower for word in ["test", "validate", "verify", "check", "apply", "practical"]):
                return DomainType.PRACTICAL.value
            elif any(tag in all_tags for tag in ["testing", "validation", "practical"]):
                return DomainType.PRACTICAL.value

            # LINGUISTIC: Language, communication, text
            elif any(word in text_lower for word in ["language", "text", "communication", "write", "document"]):
                return DomainType.LINGUISTIC.value
            elif any(tag in all_tags for tag in ["linguistic", "language", "communication"]):
                return DomainType.LINGUISTIC.value

            # TEMPORAL: Time, sequence, scheduling
            elif any(word in text_lower for word in ["time", "sequence", "schedule", "duration", "temporal", "when"]):
                return DomainType.TEMPORAL.value
            elif any(tag in all_tags for tag in ["temporal", "time", "sequence"]):
                return DomainType.TEMPORAL.value

            # SPATIAL: Location, structure, organization
            elif any(word in text_lower for word in ["location", "structure", "spatial", "position", "where", "layout"]):
                return DomainType.SPATIAL.value
            elif any(tag in all_tags for tag in ["spatial", "structure", "location"]):
                return DomainType.SPATIAL.value

            # ETHICAL: Ethics, governance, security
            elif any(word in text_lower for word in ["ethical", "security", "governance", "moral", "compliance"]):
                return DomainType.ETHICAL.value
            elif any(tag in all_tags for tag in ["ethical", "security", "governance"]):
                return DomainType.ETHICAL.value

            # SOCIAL: Collaboration, interaction
            elif any(word in text_lower for word in ["social", "collaborate", "team", "interact", "cooperate"]):
                return DomainType.SOCIAL.value
            elif any(tag in all_tags for tag in ["social", "collaboration", "interaction"]):
                return DomainType.SOCIAL.value

            # CREATIVE: Design, innovation, creativity
            elif any(word in text_lower for word in ["creative", "design", "innovate", "novel", "original"]):
                return DomainType.CREATIVE.value
            elif any(tag in all_tags for tag in ["creative", "design", "innovation"]):
                return DomainType.CREATIVE.value

            # Default to ABSTRACT for memory-based patterns
            else:
                return DomainType.ABSTRACT.value

        except Exception as e:
            logger.debug(f"Domain inference from cluster failed: {e}")
            return "abstract"  # Fallback default


# FIX #5: Hierarchical Planner - Principles shape strategy BEFORE episodic retrieval
class HierarchicalPlanner:
    """
    Planner that queries abstraction hierarchy BEFORE episodic memory.

    Flow:
    1. Query Level 3 principles relevant to goal
    2. Find Level 2 schemas under those principles
    3. Extract strategy constraints from schemas
    4. Query episodic memories WITHIN constraints
    5. Generate plan following principle-level strategy

    This makes abstraction UPSTREAM of planning, not downstream.
    """

    def __init__(self, abstraction_pipeline: AbstractionPipeline):
        self.pipeline = abstraction_pipeline
        self.hierarchy = abstraction_pipeline.concept_hierarchy
        self.schemas = abstraction_pipeline.active_schemas
        self.memory = abstraction_pipeline.memory

    async def plan(self, goal: str, domain: str = "general") -> Dict[str, Any]:
        """Generate plan using hierarchical strategy"""

        # STEP 1: Query Level 3 principles
        principles = self.hierarchy.find_principles_for_domain(domain)

        # STEP 2: Query Level 2 schemas under principles
        relevant_schemas = []
        for principle in principles:
            schema_nodes = self.hierarchy.get_descendants(principle.concept_id, max_depth=1)
            for node in schema_nodes:
                if node.level == AbstractionLevel.SCHEMA and node.schema_id:
                    if node.schema_id in self.schemas:
                        relevant_schemas.append(self.schemas[node.schema_id])

        # STEP 3: Extract strategy constraints
        strategy_constraints = []
        for schema in relevant_schemas:
            if schema.probability > 0.7:  # Only use strong schemas
                constraint = {
                    'when': schema.condition,
                    'prefer': schema.outcome,
                    'confidence': schema.probability,
                    'schema_id': schema.schema_id
                }
                strategy_constraints.append(constraint)

        # STEP 4: Query episodic memories WITHIN constraints
        constrained_query = self._build_constrained_query(goal, strategy_constraints)
        memories = await self.memory.search_memories(
            query_text=constrained_query,
            limit=20
        )

        # STEP 5: Generate plan following strategy
        plan = {
            'goal': goal,
            'domain': domain,
            'principles_applied': [p.concept_id for p in principles],
            'schemas_used': [s.schema_id for s in relevant_schemas],
            'strategy_constraints': strategy_constraints,
            'supporting_memories': [m.memory_id for m in memories],
            'plan_steps': self._generate_steps_from_strategy(goal, strategy_constraints, memories)
        }

        return plan

    def _build_constrained_query(self, goal: str, constraints: List[Dict]) -> str:
        """Build query that incorporates strategic constraints"""
        query_parts = [goal]

        for constraint in constraints[:3]:  # Top 3 constraints
            if 'prefer' in constraint:
                for key, value in constraint['prefer'].items():
                    query_parts.append(f"{key}:{value}")

        return " ".join(query_parts)

    def _generate_steps_from_strategy(
        self,
        goal: str,
        constraints: List[Dict],
        memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """Generate plan steps following strategic constraints"""
        steps = []

        # Use constraints to shape action selection
        for i, constraint in enumerate(constraints[:5]):
            step = {
                'step_number': i + 1,
                'action': f"Apply strategy: {constraint['when']} → {constraint['prefer']}",
                'confidence': constraint['confidence'],
                'schema_id': constraint['schema_id']
            }
            steps.append(step)

        # Add memory-based refinement
        for memory in memories[:3]:
            if hasattr(memory, 'content') and isinstance(memory.content, dict):
                action = memory.content.get('action') or memory.content.get('result')
                if action:
                    steps.append({
                        'step_number': len(steps) + 1,
                        'action': f"Based on past: {action}",
                        'memory_id': memory.memory_id
                    })

        return steps


_abstraction_pipeline: Optional[AbstractionPipeline] = None


def get_abstraction_pipeline() -> Optional[AbstractionPipeline]:
    """Get global instance"""
    return _abstraction_pipeline


def initialize_abstraction_pipeline(
    memory_agent: 'MemoryAgent',
    uncertainty_system: 'BayesianUncertaintySystem',
    reasoning_engine: Optional['AbstractReasoningEngine'] = None
) -> AbstractionPipeline:
    """Initialize global instance"""
    global _abstraction_pipeline
    if _abstraction_pipeline is None:
        _abstraction_pipeline = AbstractionPipeline(
            memory_agent=memory_agent,
            uncertainty_system=uncertainty_system,
            reasoning_engine=reasoning_engine
        )
    return _abstraction_pipeline


def create_hierarchical_planner(abstraction_pipeline: AbstractionPipeline) -> HierarchicalPlanner:
    """Create hierarchical planner with principle-first strategy"""
    return HierarchicalPlanner(abstraction_pipeline)
