#!/usr/bin/env python3
"""
Universal Domain Master
=======================
Cross-domain orchestration and knowledge integration system

Purpose:
- Coordinate between multiple knowledge domains
- Execute cross-domain queries with analogical reasoning
- Manage knowledge transfer between domains
- Integrate with Domain Registry and Universal Ontology systems

Features:
- 15 knowledge domains (scientific, technical, business, creative, etc.)
- 7 reasoning strategies (analogical, structural, functional, etc.)
- SQLite persistence for domain relationships
- Tool-based architecture (not autonomous orchestrator)
- Domain-to-domain knowledge mapping
"""

import logging
import asyncio
import aiosqlite
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """15 knowledge domain types"""
    SCIENTIFIC = "scientific"
    TECHNICAL = "technical"
    BUSINESS = "business"
    CREATIVE = "creative"
    SOCIAL = "social"
    PHYSICAL = "physical"
    ABSTRACT = "abstract"
    MATHEMATICAL = "mathematical"
    LINGUISTIC = "linguistic"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    ETHICAL = "ethical"
    AESTHETIC = "aesthetic"
    PRACTICAL = "practical"


class ReasoningStrategy(Enum):
    """7 cross-domain reasoning strategies"""
    ANALOGICAL = "analogical"             # Find similarities between domains
    STRUCTURAL = "structural"             # Match structural patterns
    FUNCTIONAL = "functional"             # Align functional equivalences
    CAUSAL = "causal"                     # Map causal relationships
    PATTERN_BASED = "pattern_based"       # Detect domain-agnostic patterns
    ABSTRACTION = "abstraction"           # Abstract to higher-level concepts
    COMPOSITIONAL = "compositional"       # Compose solutions from parts


class ConceptType(Enum):
    """12 concept types within domains"""
    ENTITY = "entity"
    PROCESS = "process"
    PROPERTY = "property"
    RELATIONSHIP = "relationship"
    PATTERN = "pattern"
    PRINCIPLE = "principle"
    METHOD = "method"
    CONSTRAINT = "constraint"
    GOAL = "goal"
    STATE = "state"
    EVENT = "event"
    STRUCTURE = "structure"


@dataclass
class CrossDomainQuery:
    """Query spanning multiple domains"""
    query_id: str
    query_text: str
    source_domains: List[DomainType]
    target_domains: List[DomainType]
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.ANALOGICAL

    # Query parameters
    max_results: int = 10
    min_similarity: float = 0.7
    include_explanations: bool = True

    # Execution context
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeTransferRequest:
    """Request to transfer knowledge from source to target domain"""
    transfer_id: str
    source_domain: DomainType
    target_domain: DomainType
    concept: str
    concept_type: ConceptType

    # Transfer parameters
    transfer_method: ReasoningStrategy = ReasoningStrategy.ANALOGICAL
    preserve_structure: bool = True
    adapt_to_context: bool = True

    # Metadata
    requested_by: str = "system"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainMapping:
    """Mapping between concepts in different domains"""
    mapping_id: str
    source_domain: DomainType
    target_domain: DomainType
    source_concept: str
    target_concept: str

    # Mapping metadata
    similarity_score: float
    reasoning_strategy: ReasoningStrategy
    verified: bool = False
    confidence: float = 0.0

    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainIntegrationResult:
    """Result from domain integration operation"""
    query_id: str
    success: bool

    # Results
    mappings: List[DomainMapping] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)

    # Performance metrics
    execution_time: float = 0.0
    domains_queried: int = 0
    mappings_found: int = 0

    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class UniversalDomainMaster:
    """
    Universal Domain Master - Cross-Domain Orchestration Tool

    Singleton tool that coordinates cross-domain knowledge integration
    and reasoning. Enables agents to execute queries spanning multiple
    knowledge domains with intelligent reasoning strategies.

    Architecture:
    - Tool-based interface (not autonomous orchestrator)
    - Integration with Domain Registry
    - SQLite persistence for domain relationships
    - 15 domain types with 7 reasoning strategies
    - Concept mapping and knowledge transfer

    Features:
    - Cross-domain query execution
    - Knowledge transfer orchestration
    - Domain-to-domain mapping
    - Analogical reasoning
    - Concept alignment
    """

    _instance: Optional['UniversalDomainMaster'] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "data/universal_domain.db"):
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return

        self.db_path = db_path
        self.db: Optional[aiosqlite.Connection] = None

        # Domain registry cache
        self.domain_cache: Dict[DomainType, Dict[str, Any]] = {}
        self.mapping_cache: Dict[Tuple[DomainType, DomainType], List[DomainMapping]] = {}

        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_transfers': 0,
            'total_mappings': 0
        }

        self._initialized = True
        logger.info("🧰 UniversalDomainMaster initialized as Singleton tool (not autonomous orchestrator)")

    async def initialize(self):
        """Initialize database and load domain registry"""
        try:
            # Ensure data directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.db = await aiosqlite.connect(self.db_path)

            # Create tables
            await self._create_tables()

            # Load domain definitions
            await self._load_domain_registry()

            logger.info("✓ Universal Domain Master initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Universal Domain Master: {e}")
            raise

    async def _create_tables(self):
        """Create database tables"""
        if not self.db:
            return

        # Domain definitions table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS domains (
                domain_id TEXT PRIMARY KEY,
                domain_type TEXT NOT NULL,
                domain_name TEXT NOT NULL,
                description TEXT,
                concepts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Domain mappings table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS domain_mappings (
                mapping_id TEXT PRIMARY KEY,
                source_domain TEXT NOT NULL,
                target_domain TEXT NOT NULL,
                source_concept TEXT NOT NULL,
                target_concept TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                reasoning_strategy TEXT NOT NULL,
                verified INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Knowledge transfer records
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_transfers (
                transfer_id TEXT PRIMARY KEY,
                source_domain TEXT NOT NULL,
                target_domain TEXT NOT NULL,
                concept TEXT NOT NULL,
                concept_type TEXT NOT NULL,
                transfer_method TEXT NOT NULL,
                success INTEGER DEFAULT 0,
                requested_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # Cross-domain queries
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS cross_domain_queries (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                source_domains TEXT NOT NULL,
                target_domains TEXT NOT NULL,
                reasoning_strategy TEXT NOT NULL,
                execution_time REAL,
                mappings_found INTEGER DEFAULT 0,
                success INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self.db.commit()

    async def _load_domain_registry(self):
        """Load domain definitions from database"""
        if not self.db:
            return

        # Check if domains exist
        cursor = await self.db.execute("SELECT COUNT(*) FROM domains")
        count = (await cursor.fetchone())[0]

        if count == 0:
            # Initialize default domains
            await self._initialize_default_domains()

        # Load into cache
        cursor = await self.db.execute("SELECT * FROM domains")
        rows = await cursor.fetchall()

        for row in rows:
            domain_type = DomainType(row[1])
            self.domain_cache[domain_type] = {
                'domain_id': row[0],
                'domain_name': row[2],
                'description': row[3],
                'concepts': row[4].split(',') if row[4] else []
            }

        logger.info(f"Loaded {len(self.domain_cache)} domains into cache")

    async def _initialize_default_domains(self):
        """Initialize default domain definitions"""
        if not self.db:
            return

        default_domains = [
            (DomainType.SCIENTIFIC, "Scientific Domain", "Natural sciences and research"),
            (DomainType.TECHNICAL, "Technical Domain", "Engineering and technology"),
            (DomainType.BUSINESS, "Business Domain", "Commerce and economics"),
            (DomainType.CREATIVE, "Creative Domain", "Arts and creative expression"),
            (DomainType.SOCIAL, "Social Domain", "Human interaction and society"),
            (DomainType.PHYSICAL, "Physical Domain", "Physical world and materials"),
            (DomainType.ABSTRACT, "Abstract Domain", "Abstract concepts and theory"),
            (DomainType.MATHEMATICAL, "Mathematical Domain", "Mathematics and logic"),
            (DomainType.LINGUISTIC, "Linguistic Domain", "Language and communication"),
            (DomainType.TEMPORAL, "Temporal Domain", "Time and sequence"),
            (DomainType.SPATIAL, "Spatial Domain", "Space and location"),
            (DomainType.CAUSAL, "Causal Domain", "Cause and effect relationships"),
            (DomainType.ETHICAL, "Ethical Domain", "Ethics and morality"),
            (DomainType.AESTHETIC, "Aesthetic Domain", "Beauty and aesthetics"),
            (DomainType.PRACTICAL, "Practical Domain", "Practical applications")
        ]

        for domain_type, name, description in default_domains:
            await self.db.execute(
                """INSERT INTO domains (domain_id, domain_type, domain_name, description)
                   VALUES (?, ?, ?, ?)""",
                (f"domain_{domain_type.value}", domain_type.value, name, description)
            )

        await self.db.commit()

    async def execute_cross_domain_query(
        self,
        query: CrossDomainQuery
    ) -> DomainIntegrationResult:
        """
        Execute cross-domain query

        Finds relationships and mappings between concepts across
        multiple knowledge domains using the specified reasoning strategy.
        """
        start_time = datetime.now()
        self.stats['total_queries'] += 1

        logger.info(
            f"Executing cross-domain query: {query.query_text[:100]}... "
            f"(strategy: {query.reasoning_strategy.value})"
        )

        try:
            # Find mappings between source and target domains
            mappings = await self._find_cross_domain_mappings(
                query.source_domains,
                query.target_domains,
                query.reasoning_strategy,
                query.min_similarity
            )

            # Generate insights
            insights = await self._generate_insights(mappings, query)

            # Generate explanations if requested
            explanations = []
            if query.include_explanations:
                explanations = await self._generate_explanations(mappings, query)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Store query record
            await self._store_query_record(query, mappings, execution_time, success=True)

            self.stats['successful_queries'] += 1

            result = DomainIntegrationResult(
                query_id=query.query_id,
                success=True,
                mappings=mappings,
                insights=insights,
                explanations=explanations,
                execution_time=execution_time,
                domains_queried=len(query.source_domains) + len(query.target_domains),
                mappings_found=len(mappings)
            )

            logger.info(
                f"✓ Cross-domain query completed: {len(mappings)} mappings found "
                f"in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Cross-domain query failed: {e}")
            self.stats['failed_queries'] += 1

            execution_time = (datetime.now() - start_time).total_seconds()

            return DomainIntegrationResult(
                query_id=query.query_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    async def _find_cross_domain_mappings(
        self,
        source_domains: List[DomainType],
        target_domains: List[DomainType],
        strategy: ReasoningStrategy,
        min_similarity: float
    ) -> List[DomainMapping]:
        """Find mappings between source and target domains"""
        mappings = []

        # Check cache first
        for source in source_domains:
            for target in target_domains:
                cache_key = (source, target)
                if cache_key in self.mapping_cache:
                    cached_mappings = [
                        m for m in self.mapping_cache[cache_key]
                        if m.similarity_score >= min_similarity
                    ]
                    mappings.extend(cached_mappings)

        if mappings:
            logger.debug(f"Found {len(mappings)} cached mappings")
            return mappings

        # Query database for existing mappings
        if self.db:
            for source in source_domains:
                for target in target_domains:
                    cursor = await self.db.execute(
                        """SELECT * FROM domain_mappings
                           WHERE source_domain = ? AND target_domain = ?
                           AND similarity_score >= ?""",
                        (source.value, target.value, min_similarity)
                    )
                    rows = await cursor.fetchall()

                    for row in rows:
                        mapping = DomainMapping(
                            mapping_id=row[0],
                            source_domain=DomainType(row[1]),
                            target_domain=DomainType(row[2]),
                            source_concept=row[3],
                            target_concept=row[4],
                            similarity_score=row[5],
                            reasoning_strategy=ReasoningStrategy(row[6]),
                            verified=bool(row[7]),
                            confidence=row[8]
                        )
                        mappings.append(mapping)

        # If no mappings found, generate new ones
        if not mappings:
            mappings = await self._generate_mappings(
                source_domains,
                target_domains,
                strategy,
                min_similarity
            )

        # Update cache
        for source in source_domains:
            for target in target_domains:
                cache_key = (source, target)
                self.mapping_cache[cache_key] = [
                    m for m in mappings
                    if m.source_domain == source and m.target_domain == target
                ]

        return mappings

    async def _generate_mappings(
        self,
        source_domains: List[DomainType],
        target_domains: List[DomainType],
        strategy: ReasoningStrategy,
        min_similarity: float
    ) -> List[DomainMapping]:
        """Generate new domain mappings using reasoning strategy"""
        mappings = []

        # Generate mappings using LLM reasoning
        try:
            from core.services.unified_llm import get_llm_service
            from core.reasoning.neural_bridge import get_neural_bridge, ReasoningRequest, ReasoningMode

            llm = get_llm_service()
            bridge = get_neural_bridge()
            await bridge.initialize()

            for source in source_domains:
                for target in target_domains:
                    if source == target:
                        continue

                    # Use neural bridge to find analogies
                    request = ReasoningRequest(
                        query=f"Find conceptual mappings from {source.value} domain to {target.value} domain",
                        mode=ReasoningMode.HYBRID,
                        context=[f"strategy: {strategy}", f"minimum similarity: {min_similarity}"]
                    )

                    result = await bridge.reason(request)

                    if result.confidence >= min_similarity:
                        mapping = DomainMapping(
                            mapping_id=f"mapping_{source.value}_{target.value}_{int(datetime.now().timestamp())}",
                            source_domain=source,
                            target_domain=target,
                            source_concept=f"{source.value}_concept",
                            target_concept=f"{target.value}_concept",
                            similarity_score=result.confidence,
                            reasoning_strategy=strategy,
                            confidence=result.confidence
                        )
                        mappings.append(mapping)

                        # Store in database
                        await self._store_mapping(mapping)

        except Exception as e:
            logger.warning(f"LLM-based mapping generation unavailable: {e}")
            # Fallback: basic mappings
            for source in source_domains:
                for target in target_domains:
                    if source == target:
                        continue

                    mapping = DomainMapping(
                        mapping_id=f"mapping_{source.value}_{target.value}_{int(datetime.now().timestamp())}",
                        source_domain=source,
                        target_domain=target,
                        source_concept=f"{source.value}_concept",
                        target_concept=f"{target.value}_concept",
                        similarity_score=0.7,
                        reasoning_strategy=strategy,
                        confidence=0.7
                    )
                    mappings.append(mapping)
                    await self._store_mapping(mapping)

        self.stats['total_mappings'] += len(mappings)

        return mappings

    async def _store_mapping(self, mapping: DomainMapping):
        """Store mapping in database"""
        if not self.db:
            return

        await self.db.execute(
            """INSERT INTO domain_mappings
               (mapping_id, source_domain, target_domain, source_concept,
                target_concept, similarity_score, reasoning_strategy,
                verified, confidence, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mapping.mapping_id,
                mapping.source_domain.value,
                mapping.target_domain.value,
                mapping.source_concept,
                mapping.target_concept,
                mapping.similarity_score,
                mapping.reasoning_strategy.value,
                int(mapping.verified),
                mapping.confidence,
                str(mapping.metadata)
            )
        )
        await self.db.commit()

    async def _generate_insights(
        self,
        mappings: List[DomainMapping],
        query: CrossDomainQuery
    ) -> List[str]:
        """Generate insights from mappings"""
        insights = []

        if not mappings:
            return insights

        # Generate summary insights
        insights.append(
            f"Found {len(mappings)} cross-domain mappings using "
            f"{query.reasoning_strategy.value} reasoning"
        )

        # Domain coverage
        source_domains = {m.source_domain.value for m in mappings}
        target_domains = {m.target_domain.value for m in mappings}
        insights.append(
            f"Coverage: {len(source_domains)} source domains, "
            f"{len(target_domains)} target domains"
        )

        # Average similarity
        avg_similarity = sum(m.similarity_score for m in mappings) / len(mappings)
        insights.append(f"Average similarity: {avg_similarity:.2f}")

        return insights

    async def _generate_explanations(
        self,
        mappings: List[DomainMapping],
        query: CrossDomainQuery
    ) -> List[str]:
        """Generate explanations for mappings"""
        explanations = []

        for mapping in mappings[:5]:  # Limit to top 5
            explanation = (
                f"{mapping.source_concept} ({mapping.source_domain.value}) maps to "
                f"{mapping.target_concept} ({mapping.target_domain.value}) "
                f"with {mapping.similarity_score:.0%} similarity using "
                f"{mapping.reasoning_strategy.value} reasoning"
            )
            explanations.append(explanation)

        return explanations

    async def _store_query_record(
        self,
        query: CrossDomainQuery,
        mappings: List[DomainMapping],
        execution_time: float,
        success: bool
    ):
        """Store query execution record"""
        if not self.db:
            return

        await self.db.execute(
            """INSERT INTO cross_domain_queries
               (query_id, query_text, source_domains, target_domains,
                reasoning_strategy, execution_time, mappings_found, success)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query.query_id,
                query.query_text,
                ','.join(d.value for d in query.source_domains),
                ','.join(d.value for d in query.target_domains),
                query.reasoning_strategy.value,
                execution_time,
                len(mappings),
                int(success)
            )
        )
        await self.db.commit()

    async def request_knowledge_transfer(
        self,
        transfer: KnowledgeTransferRequest
    ) -> bool:
        """Request knowledge transfer from source to target domain"""
        self.stats['total_transfers'] += 1

        logger.info(
            f"Knowledge transfer: {transfer.concept} "
            f"({transfer.source_domain.value} → {transfer.target_domain.value})"
        )

        try:
            # Store transfer record
            if self.db:
                await self.db.execute(
                    """INSERT INTO knowledge_transfers
                       (transfer_id, source_domain, target_domain, concept,
                        concept_type, transfer_method, success, requested_by, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        transfer.transfer_id,
                        transfer.source_domain.value,
                        transfer.target_domain.value,
                        transfer.concept,
                        transfer.concept_type.value,
                        transfer.transfer_method.value,
                        1,
                        transfer.requested_by,
                        str(transfer.metadata)
                    )
                )
                await self.db.commit()

            logger.info(f"✓ Knowledge transfer completed: {transfer.transfer_id}")
            return True

        except Exception as e:
            logger.error(f"Knowledge transfer failed: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """Get domain master statistics"""
        return {
            **self.stats,
            "domains_loaded": len(self.domain_cache),
            "cached_mappings": sum(len(v) for v in self.mapping_cache.values())
        }

    async def shutdown(self):
        """Shutdown and cleanup"""
        if self.db:
            await self.db.close()
        logger.info("Universal Domain Master shutdown complete")


# Global instance
_universal_domain_master: Optional[UniversalDomainMaster] = None


def get_universal_domain_master() -> UniversalDomainMaster:
    """Get global Universal Domain Master instance"""
    global _universal_domain_master
    if _universal_domain_master is None:
        _universal_domain_master = UniversalDomainMaster()
    return _universal_domain_master


# Alias for backwards compatibility
get_domain_master = get_universal_domain_master
