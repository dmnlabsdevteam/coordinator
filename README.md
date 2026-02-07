# TorinAI Autonomous Coordinator - Core Modules

**Research Snapshot**: Domain-Aware Autonomous Cognition System

This repository contains a snapshot of TorinAI's autonomous coordination system, showcasing the integration of memory, reasoning, domain expertise, and governance systems into a unified cognitive architecture.

---

## 🎯 Purpose

This is a **collaborative reference implementation** demonstrating:
- Autonomous task coordination with intrinsic motivation
- Domain-aware learning and cross-domain knowledge transfer
- Persistent cognition with continuous background loops
- Governance feedback loops preventing repetitive failures
- Hierarchical abstraction with temporal decay

This is **not an actively maintained repo** - it's a snapshot for research collaboration and architectural review.

---

## 📁 Repository Structure

```
coordinator-export/
├── agents/
│   ├── autonomous_coordinator.py   # Main autonomous orchestrator
│   ├── intrinsic_motivation.py     # 7-dimensional motivation system
│   ├── memory_agent.py              # Memory coordination & persistence
│   ├── governance_agent.py          # Governance enforcement
│   ├── runtime_governance.py        # Runtime governance validation
│   └── shared_types.py              # Shared type definitions
├── reasoning/
│   ├── bayesian_uncertainty.py      # Bayesian belief system with temporal decay
│   └── hierarchical_abstraction.py  # Episodic → Pattern → Schema → Principle
├── integration/
│   └── universal_domain_master.py   # 15-domain cross-domain reasoning
└── README.md
```

---

## 🏗️ Architecture Overview

### Core Components

#### 1. **Autonomous Coordinator** (`agents/autonomous_coordinator.py`)
The main orchestrator that runs continuous autonomous cycles:

- **Perception Phase**: Gathers system state, recent feedback, causal analysis
- **Planning Phase**: Generates goals from intrinsic motivation + extrinsic tasks
- **Execution Phase**: Executes tasks with LLM + tool integration
- **Reflection Phase**: Stores outcomes as META memories for learning

**Key Features**:
- Domain-aware task inference (maps to 15 Universal Domain types)
- Governance feedback integration (queries META memory for blocked patterns)
- Cross-domain insight generation during goal formation
- Persistent task execution history in MySQL

**Recent Wiring**: Now queries Universal Domain Master during goal generation to provide domain competency profiles and cross-domain transfer opportunities.

---

#### 2. **Intrinsic Motivation System** (`agents/intrinsic_motivation.py`)
7-dimensional motivation engine influencing 60% of autonomous decisions:

**Dimensions**:
1. **Curiosity** (1.2x weight) - Novel exploration
2. **Competence** (0.9x) - Skill improvement
3. **Novelty** (0.85x) - New experiences
4. **Mastery** (0.7x) - Deep understanding
5. **Autonomy** (1.0x) - Self-direction
6. **Social** (0.9x) - Collaboration
7. **Impact** (0.8x) - Meaningful change

**Key Features**:
- Context-driven goal generation (not template-based)
- Governance block querying to avoid repetition
- Domain performance statistics via META memory
- **NEW**: `get_skill_recommendations()` - ranks domains by learning potential

**Skill Recommendation Algorithm**:
```python
score = (
    meta_success_rate * 0.40 +      # META memory (rewards 50-80% success)
    belief_confidence * 0.25 +       # Bayesian beliefs (rewards moderate confidence)
    abstraction_coverage * 0.20 +    # Schema count (rewards 2-8 schemas)
    transfer_potential * 0.15        # Cross-domain mappings
)
```

**Recent Wiring**: Integrates with Universal Domain Master, Bayesian system, and HierarchicalAbstraction to score all 15 domains and recommend learning zones.

---

#### 3. **Memory Agent** (`agents/memory_agent.py`)
Unified memory coordination with persistent cognition:

**Architecture**:
- Hot/Cold tier MySQL storage (0-60 days hot, 60+ days cold)
- Memory types: EPISODIC, SEMANTIC, PROCEDURAL, META
- Continuous background loops (HealthMonitor pattern)

**Background Loops**:
1. **Maintenance Loop** (1 hour): Consolidation, hot→cold migration, cleanup, decay
2. **Abstraction Loop** (4 hours): Pattern extraction, schema formation
3. **Reflection Loop** (24 hours): Belief consistency, domain volatility updates

**Recent Wiring**: Queries include domain filters, enabling domain-specific retrieval for competency tracking.

---

#### 4. **Hierarchical Abstraction** (`reasoning/hierarchical_abstraction.py`)
4-level abstraction pipeline with architectural fixes:

**Levels**:
- **Level 0**: Episodic memories (raw experiences)
- **Level 1**: Patterns (repeated structures)
- **Level 2**: Schemas (probabilistic if-then rules with counterexamples)
- **Level 3**: Principles (meta-patterns across domains)

**Architectural Fixes**:
1. ✅ **Semantic Overreach Protection**: 4-layer validation (50% word overlap + domain + embedding@0.6 + ontology)
2. ✅ **Feedback Loop Damping**: Cumulative caps (50% retrieval boost max, 30% prior adjustment max)
3. ✅ **Full-Spectrum Decay**: Decays probability, attention, retrieval boost, and prior adjustments
4. ✅ **Counterfactual Stress Testing**: Condition-flip and outcome-flip scenarios
5. ✅ **Hierarchical Planner**: Principles query BEFORE episodic memories

**Recent Wiring**:
- Schemas now tagged with domain labels via `_infer_domain_from_cluster()`
- Domain inference uses keyword + tag analysis mapped to 15 DomainTypes
- Principles extraction groups schemas by domain

---

#### 5. **Bayesian Uncertainty System** (`reasoning/bayesian_uncertainty.py`)
Probabilistic belief tracking with temporal decay:

**Key Features**:
- Temporal decay prevents epistemic ossification: `P(H)_t+1 = P(H)_t + (0.5 - P(H)_t) * (1 - exp(-λΔt))`
- Domain-adaptive decay rates based on volatility
- Belief dependency graphs with constraint propagation (IMPLIES, CONTRADICTS, SUPPORTS)
- Autonomous reflection methods for background loops

**Decay Mechanisms**:
- λ (decay rate) adapts per domain based on regime shifts
- Beliefs decay toward 0.5 (maximum uncertainty) over time
- Evidence updates reset decay timer

---

#### 6. **Universal Domain Master** (`integration/universal_domain_master.py`)
Cross-domain orchestration tool with 15 knowledge domains:

**Domains**:
SCIENTIFIC, TECHNICAL, BUSINESS, CREATIVE, SOCIAL, PHYSICAL, ABSTRACT, MATHEMATICAL, LINGUISTIC, TEMPORAL, SPATIAL, CAUSAL, ETHICAL, AESTHETIC, PRACTICAL

**Features**:
- 7 reasoning strategies (analogical, structural, functional, causal, pattern-based, abstraction, compositional)
- Cross-domain query execution with mapping generation
- Knowledge transfer orchestration
- SQLite persistence for domain relationships

**Recent Wiring**: Previously orphaned, now integrated into:
- Skill recommendations (mapping cache queried for transfer potential)
- Goal generation (domain statistics exposed to coordinator)
- Task/schema inference (DomainType enum used throughout)

---

#### 7. **Governance Agents** (`agents/governance_agent.py`, `agents/runtime_governance.py`)
Dual-layer governance enforcement:

**Governance Agent**:
- Constitutional law enforcement (5 core principles + 12 derived laws)
- Action validation against governance rules
- Violation tracking and reporting

**Runtime Governance**:
- Real-time action validation during execution
- Security integration (validates with SecurityController)
- Blocks stored as META memories for learning

**Recent Wiring**:
- Blocks stored with domain tags
- IntrinsicMotivation queries META memories before goal generation to avoid blocked patterns
- Prevents "groundhog day" problem (repeating failed actions)

---

## 🔗 System Integration Map

### Complete Flow: Domain → Memory → Abstraction → Motivation → Action

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS COORDINATOR                        │
│  • Orchestrates continuous autonomous cycles                    │
│  • Maps tasks to DomainTypes (15 domains)                       │
│  • Stores outcomes as META memories with domain tags            │
└──────────────┬──────────────────────────────────┬───────────────┘
               │                                  │
               ▼                                  ▼
    ┌──────────────────────┐          ┌──────────────────────┐
    │  INTRINSIC MOTIVATION│          │   MEMORY AGENT       │
    │  • 7 dimensions      │◄─────────┤  • Hot/Cold tiers    │
    │  • Skill recs (NEW)  │          │  • 3 background loops│
    │  • Goal generation   │          │  • Domain filtering  │
    └──────────┬───────────┘          └──────────┬───────────┘
               │                                  │
               │ Queries domains                  │ Provides memories
               ▼                                  ▼
    ┌──────────────────────┐          ┌──────────────────────┐
    │ UNIVERSAL DOMAIN     │          │  HIERARCHICAL        │
    │ MASTER               │          │  ABSTRACTION         │
    │ • 15 domains         │          │  • 4 levels          │
    │ • Cross-domain maps  │          │  • Domain-tagged     │
    │ • Transfer potential │◄─────────┤    schemas (NEW)     │
    └──────────────────────┘          └──────────┬───────────┘
               │                                  │
               │ Competency profile               │ Abstractions
               ▼                                  ▼
    ┌──────────────────────────────────────────────────────────┐
    │                    BAYESIAN UNCERTAINTY                   │
    │  • Temporal decay per domain                             │
    │  • Belief dependency graphs                              │
    │  • Domain volatility tracking                            │
    └──────────────────────────────────────────────────────────┘
```

### Data Flow: How Domain Awareness Propagates

1. **Task Execution** → Coordinator infers domain from task description
2. **META Memory Storage** → Outcome stored with `domain_{type}` tag
3. **Memory Clustering** → Abstraction system clusters memories by similarity
4. **Schema Formation** → Cluster domain inferred, schema tagged with domain
5. **Principle Extraction** → Schemas grouped by domain, principles per domain
6. **Skill Recommendations** → Queries META memories by domain for success rates
7. **Goal Generation** → Coordinator provides domain insights to Singleton LLM

---

## 🚀 Key Innovations

### 1. **Governance Feedback Loop**
**Problem**: Autonomous system repeats blocked actions ("groundhog day")

**Solution**:
```python
# During goal generation (intrinsic_motivation.py)
governance_constraints = await self._query_governance_blocks()
# Returns: ["Avoid: security audit (blocked: unauthorized)", ...]

# Context provided to LLM includes:
"""
GOVERNANCE CONSTRAINTS (avoid these patterns):
  - Avoid: modifying production database (blocked: security_validation)
  - Avoid: external API calls without auth (blocked: governance_law)
"""
```

### 2. **Domain-Aware Skill Recommendations**
**Problem**: No systematic way to identify learning opportunities

**Solution**: 4-factor scoring across 15 domains
- Identifies "learning zones" (50-80% success rate)
- Rewards moderate confidence (room to grow)
- Tracks abstraction coverage (2-8 schemas = optimal)
- Considers cross-domain transfer potential

### 3. **Persistent Cognition Architecture**
**Problem**: Memory system was reactive ("on-call"), not autonomous

**Solution**: 3 continuous background loops
- Maintenance (1hr): Consolidation, migration, cleanup, decay
- Abstraction (4hr): Pattern extraction, schema formation
- Reflection (24hr): Belief consistency, domain volatility

### 4. **Full-Spectrum Temporal Decay**
**Problem**: Only decaying probability leaves "zombie beliefs" with high attention weights

**Solution**: Decay across 4 dimensions
```python
decayed_prob = current + (0.5 - current) * decay_factor
decayed_attention = current + (1.0 - current) * decay_factor
decayed_boost = current + (1.0 - current) * decay_factor
decayed_prior_adj = current * (1 - decay_factor * 0.5)
```

### 5. **Hierarchical Planning**
**Problem**: Abstractions formed AFTER planning (too late to influence strategy)

**Solution**: Query hierarchy BEFORE episodic memories
1. Query Level 3 principles relevant to goal
2. Find Level 2 schemas under principles
3. Extract strategy constraints
4. Query episodic memories WITHIN constraints
5. Generate plan following principle-level strategy

---

## 📊 Domain Integration Statistics

**15 Universal Domains**:
- Scientific, Technical, Business, Creative, Social
- Physical, Abstract, Mathematical, Linguistic, Temporal
- Spatial, Causal, Ethical, Aesthetic, Practical

**Integration Points**:
- ✅ Task → Domain inference (autonomous_coordinator.py)
- ✅ Schema → Domain tagging (hierarchical_abstraction.py)
- ✅ META memory → Domain filtering (memory_agent.py)
- ✅ Skill recommendations → Domain scoring (intrinsic_motivation.py)
- ✅ Goal generation → Domain insights (autonomous_coordinator.py)

**Cross-Domain Reasoning**:
- 7 reasoning strategies (analogical, structural, functional, causal, pattern, abstraction, compositional)
- Mapping cache tracks domain-to-domain relationships
- Transfer potential scored as 15% weight in skill recommendations

---

## 🔧 Technical Details

### Memory Architecture
- **Hot Tier**: MySQL `torinai_thinking_hot` (0-60 days)
- **Cold Tier**: MySQL `torinai_memory_cold` (60+ days)
- **Migration**: Automatic via maintenance loop
- **Decay**: Applied to importance scores: `importance * exp(-0.01 * age_days)`

### Belief System
- **Representation**: Bayesian probabilities with confidence intervals
- **Dependencies**: Graph with typed edges (IMPLIES, CONTRADICTS, SUPPORTS, COMPETES_WITH)
- **Decay**: Domain-adaptive λ based on volatility
- **Propagation**: Constraint satisfaction across belief network

### Abstraction Pipeline
- **Clustering**: Semantic similarity threshold 0.75
- **Pressure Threshold**: 5.0 (minimum for abstraction formation)
- **Schema Validation**: Requires counterexample tracking
- **Principle Formation**: Minimum 5 schemas, 3 per domain

### Intrinsic Motivation
- **Influence**: 60% of autonomous decisions
- **Calculation**: Weighted sum across 7 dimensions
- **Goal Generation**: LLM-based with system context
- **Avoidance**: Tracks last 20 goal descriptions to prevent repetition

---

## 🧪 Research Questions Addressed

1. **How to prevent epistemic ossification in long-running autonomous systems?**
   - ✅ Temporal decay with domain-adaptive rates
   - ✅ Belief dependency graphs with propagation

2. **How to avoid "groundhog day" in autonomous goal generation?**
   - ✅ Governance blocks stored as META memories
   - ✅ Queried before goal generation to avoid patterns

3. **How to integrate domain expertise into general cognition?**
   - ✅ Universal Domain Master with 15 domains
   - ✅ Cross-domain mapping and transfer potential
   - ✅ Domain-tagged schemas and META memories

4. **How to make abstractions influence planning (not just storage)?**
   - ✅ Hierarchical planner queries principles FIRST
   - ✅ Constrains episodic retrieval with abstract strategy

5. **How to prevent feedback loop explosion in self-modifying systems?**
   - ✅ Cumulative caps on retrieval boosts and prior adjustments
   - ✅ Novel evidence requirements after 5 reinforcements
   - ✅ Full-spectrum decay across all modified parameters

---

## 📚 Key Files Deep Dive

### `autonomous_coordinator.py` (5000+ lines)
**Main orchestrator** - runs continuous autonomous cycles

**Critical Methods**:
- `run()` - Main loop: perception → planning → execution → reflection
- `_store_task_outcome_meta_memory()` - Stores outcomes with domain tags
- `_store_governance_block_meta_memory()` - Tracks blocked actions
- `_infer_domain_from_task()` - Maps task descriptions to 15 DomainTypes
- `_gather_context_for_singleton()` - Builds context including domain insights

**Recent Changes**:
- Domain inference now uses Universal Domain Master's DomainType enum
- Context includes domain competency profile and transfer opportunities
- Skill recommendations integrated into goal generation prompts

---

### `intrinsic_motivation.py` (1072 lines)
**7-dimensional motivation engine** - influences 60% of autonomous decisions

**Critical Methods**:
- `calculate_motivation()` - Computes motivation across 7 dimensions
- `generate_curiosity_driven_goals()` - LLM-based goal generation with governance awareness
- `_query_governance_blocks()` - Queries META memory for patterns to avoid
- `get_skill_recommendations()` - **NEW**: Ranks domains by learning potential
- `get_domain_performance_stats()` - Queries META memory for domain success rates

**Skill Recommendation Factors**:
1. META memory success rate (40%) - rewards learning zone (50-80%)
2. Bayesian belief confidence (25%) - rewards moderate confidence
3. Abstraction coverage (20%) - rewards 2-8 schemas per domain
4. Cross-domain transfer potential (15%) - based on mapping cache

---

### `hierarchical_abstraction.py` (1400+ lines)
**4-level abstraction pipeline** - episodic → pattern → schema → principle

**Critical Methods**:
- `extract_and_apply_schema()` - Forms schemas from memory clusters
- `_extract_probabilistic_schema()` - Creates schemas with counterexamples
- `_infer_domain_from_cluster()` - **NEW**: Maps clusters to DomainTypes
- `apply_temporal_decay()` - Full-spectrum decay (prob + attention + boost + prior)
- `stress_test_schema()` - Counterfactual validation
- `_extract_principles_from_schemas()` - Groups schemas by domain

**Architectural Protections**:
- Semantic overreach: 4-layer validation (word overlap, domain, embedding, ontology)
- Feedback loops: Cumulative caps + novel evidence requirements
- Full-spectrum decay: All parameters decay, not just probability
- Stress testing: Condition-flip and outcome-flip scenarios

---

### `universal_domain_master.py` (754 lines)
**Cross-domain orchestration tool** - 15 domains, 7 reasoning strategies

**Critical Methods**:
- `execute_cross_domain_query()` - Finds mappings between domains
- `_find_cross_domain_mappings()` - Checks cache, queries DB, generates new
- `_generate_mappings()` - Uses neural bridge for analogical reasoning
- `request_knowledge_transfer()` - Orchestrates knowledge transfer
- `get_statistics()` - Returns domain master statistics

**Domain Types** (15):
SCIENTIFIC, TECHNICAL, BUSINESS, CREATIVE, SOCIAL, PHYSICAL, ABSTRACT, MATHEMATICAL, LINGUISTIC, TEMPORAL, SPATIAL, CAUSAL, ETHICAL, AESTHETIC, PRACTICAL

**Reasoning Strategies** (7):
ANALOGICAL, STRUCTURAL, FUNCTIONAL, CAUSAL, PATTERN_BASED, ABSTRACTION, COMPOSITIONAL

---

### `bayesian_uncertainty.py` (600+ lines)
**Probabilistic belief system** - temporal decay, dependency graphs

**Critical Methods**:
- `apply_temporal_decay_to_all_beliefs()` - Prevents epistemic ossification
- `check_belief_consistency()` - Validates dependency graph
- `update_domain_volatility_metrics()` - Adapts decay rates
- `add_belief_dependency()` - Creates typed edges (IMPLIES, CONTRADICTS, etc.)

**Decay Formula**:
```python
P(H)_t+1 = P(H)_t + (0.5 - P(H)_t) * (1 - exp(-λΔt))
```
Where λ is domain-adaptive: `λ = base_rate * (1 + domain_volatility)`

---

### `memory_agent.py` (1500+ lines)
**Memory coordination** - hot/cold tiers, continuous loops

**Critical Methods**:
- `start_memory_loops()` - Starts 3 background loops
- `consolidate_memories()` - Migration, cleanup, decay (1hr loop)
- `search_memories()` - Query with domain filtering
- `store_memory()` - Stores with type, importance, tags

**Background Loops**:
1. `_maintenance_loop()` - Every 1 hour
2. `_abstraction_loop()` - Every 4 hours
3. `_reflection_loop()` - Every 24 hours

---

## 🎓 Usage Notes

This is a **reference snapshot**, not a standalone system. Key dependencies:

**Required External Systems**:
- MySQL databases: `torinai_thinking_hot`, `torinai_memory_cold`
- LLM service (unified_llm.py) - not included
- Neural bridge (neural_bridge.py) - for cross-domain reasoning
- Security controller - for governance validation

**Configuration**:
- Domain master uses SQLite: `data/universal_domain.db`
- Memory agent uses MySQL with hot/cold tiers
- Motivation system saves profile: `data/motivation_profile.json`

**Not Included**:
- Database schemas
- LLM integration layer
- Security/governance rule definitions
- Tool system architecture
- Task queue implementation

---

## 📝 License & Attribution

This is a **research snapshot** from TorinAI's autonomous cognition system, shared for collaborative research purposes.

**Attribution**: Dominion Labs / TorinAI Project

**Note**: This code is provided as-is for architectural reference and research collaboration. It is not a complete, runnable system without the broader TorinAI infrastructure.

---

## 🔄 Recent Integration (January 2025)

**Wiring Completed**: Universal Domain Master → Active Cognition

Previously, the Universal Domain Master existed but was completely disconnected from the active cognitive loop. Recent integration includes:

1. **Skill Recommendations** - `get_skill_recommendations()` implemented in intrinsic_motivation.py
2. **Domain Inference** - Task and schema creation now map to 15 DomainTypes
3. **META Memory Tagging** - All outcomes tagged with `domain_{type}`
4. **Goal Generation** - Domain insights flow into autonomous decision-making
5. **Cross-Domain Transfer** - Mapping cache queried for transfer opportunities

**Result**: The cognitive loop now has domain awareness at every level: perception → memory → abstraction → planning → action.

---

## 📧 Contact

For questions about this architecture or collaboration opportunities:
- GitHub Issues: https://github.com/dmnlabsdevteam/coordinator/issues
- Organization: Dominion Labs

---

**Last Updated**: January 2025
**Version**: Research Snapshot v1.0
**Status**: Reference Implementation (Not Actively Maintained)
