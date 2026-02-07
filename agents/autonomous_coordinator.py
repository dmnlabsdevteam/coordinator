#!/usr/bin/env python3
"""
Autonomous Coordinator - Main orchestrator for the modular autonomous system
Replaces the monolithic master_autonomous_controller.py with clean coordination
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque

from .shared_types import (
    SystemMode, SystemState, Task, Goal, Plan, PerceptionData,
    TaskType, TaskStatus, Priority, TaskSource
)
from .singleton_constitution import DriftSeverity
from .perception_manager import PerceptionManager
from .planning_engine import PlanningEngine
from .execution_controller import ExecutionController
from .learning_adapter import LearningAdapter
from .intrinsic_motivation import IntrinsicMotivationSystem
from .directive_system import DirectiveSystem
from .runtime_governance import get_runtime_governance

# Core system imports using absolute paths
import sys
from pathlib import Path
# Add project root to Python path for absolute imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.memory import MemoryManager, MemoryItem, MemoryQuery, MemoryType, MemoryOperation
from core.reasoning import (
    AbstractReasoningEngine, ReasoningContext, ReasoningType,
    create_abstract_reasoning_engine, AdvancedProofEngine
)
from core.learning import UnifiedLearningSystem
from core.intelligence import PredictiveIntelligenceSystem, PredictionDomain, PredictionHorizon
from core.health.system_watchdog import SystemWatchdog
from core.database.logging_database import LoggingDatabase
from core.tools.tool_registry import ToolResult
try:
    from core.monitoring.resource_config import TORIN_RESOURCE_LIMITS
except Exception:
    TORIN_RESOURCE_LIMITS = None
import uuid
from dotenv import load_dotenv

# Load environment variables from .env.production
env_file = Path(__file__).parent.parent.parent.parent / ".env.production"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback to .env if .env.production doesn't exist
    env_file_fallback = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file_fallback.exists():
        load_dotenv(env_file_fallback)

# Initialize logger first
logger = logging.getLogger(__name__)

# Security and monitoring integration
try:
    from core.security import SecurityController, get_security_controller
    SECURITY_AVAILABLE = True
except ImportError:
    SecurityController = None
    SECURITY_AVAILABLE = False
    logger.warning("Security system not available")

try:
    from core.health.monitoring_coordinator import MonitoringCoordinator
    MONITORING_AVAILABLE = True
except ImportError:
    MonitoringCoordinator = None
    MONITORING_AVAILABLE = False
    logger.warning("Monitoring coordinator not available")

# Domain system imports for cross-domain reasoning
from core.domain import DomainRegistry, UniversalOntology, CrossDomainReasoner
from core.integration.universal_domain_master import UniversalDomainMaster, CrossDomainQuery, DomainType, ReasoningStrategy

# Import Torin (the LLM brain) - CRITICAL INTEGRATION
from core.services.unified_llm import get_llm_service, LLMRequest, LLMResponse
from core.utils.notification_publisher import publish_notification
from core.integration.slack_notifier import get_slack_notifier


class AutonomousCoordinator:
    """
    Main coordinator that orchestrates perception, planning, execution, and learning
    Enhanced with cross-domain reasoning and predictive intelligence capabilities
    
    The coordinator is the interface - the brain (torin_brain) is the SOURCE of intelligence.
    All decisions, reasoning, and coordination flow through Torin's consciousness.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, torin_brain=None):
        self.config = config or {}
        self.active = False
        
        # THE SOURCE - Torin's brain is REQUIRED for conscious coordination
        if torin_brain is None:
            raise ValueError("AutonomousCoordinator requires torin_brain - the Singleton's consciousness is mandatory")
        self.torin_brain = torin_brain
        self.llm = torin_brain  # Initialize llm attribute with torin_brain

        # System state
        self.system_state = SystemState()
        self.coordination_cycle_interval = self.config.get("cycle_interval", 5.0)  # seconds
        
        # Initialize core modules
        self.perception = PerceptionManager(self.config.get("perception", {}))
        self.planning = PlanningEngine(self.config.get("planning", {}))
        self.execution = ExecutionController(self.config.get("execution", {}))
        self.learning = LearningAdapter(self.config.get("learning", {}))
        self.intrinsic_motivation = IntrinsicMotivationSystem(self.config.get("intrinsic_motivation", {}))

        # Directive System - High-level guidance for the Singleton
        self.directive_system = DirectiveSystem()

        # Runtime Governance - Validates critical decisions against governance laws
        self.runtime_governance = get_runtime_governance()

        # Singleton Constitution - Tracks compliance with the 5 governance laws
        from .singleton_constitution import get_singleton_constitution
        self.constitution = get_singleton_constitution()
        logger.info("📜 Singleton Constitution tracker initialized (using global singleton)")

        # Phase 2: Multi-level safety prompts for long-horizon planning protection
        from core.safety import MultiLevelSafetyPrompts
        self.safety_prompts = MultiLevelSafetyPrompts()
        logger.info("🛡️ Multi-level safety prompts initialized")

        # === EVENT-DRIVEN TASK EXECUTION ===
        # NEW: Event-driven task queue system (replaces hardcoded loops)
        from core.agents.autonomous.task_queue import TaskQueue
        from core.agents.autonomous.general_purpose_executor import GeneralPurposeExecutor
        from core.agents.autonomous.success_validator import SuccessValidator

        self.task_queue = TaskQueue()
        self.executor = GeneralPurposeExecutor(torin_brain)
        self.validator = SuccessValidator()
        self._idle_count = 0

        # Extrinsic task system (disabled by default - enable via config)
        self.enable_extrinsic_tasks = self.config.get("enable_extrinsic_tasks", False)
        self.extrinsic_manager = None
        if self.enable_extrinsic_tasks:
            from core.agents.autonomous.extrinsic_task_manager import ExtrinsicTaskManager
            self.extrinsic_manager = ExtrinsicTaskManager(self.task_queue)
            logger.info("🎯 Event-driven task execution system initialized (extrinsic tasks ENABLED)")
        else:
            logger.info("🎯 Event-driven task execution system initialized (extrinsic tasks DISABLED)")
        
        # Memory system - can be provided or created (will be initialized in async initialize() method)
        if 'memory' in self.config and not isinstance(self.config['memory'], dict):
            # Memory system instance provided directly
            self.memory = self.config['memory']
        else:
            # Will be initialized asynchronously in initialize() method
            self.memory = None

        # === MEMORY QUERY AGENTS ===
        # Specialized agents for querying and summarizing different memory systems
        # MySQL is PRIMARY storage (hot/cold tiers)
        self.mysql_memory_agent = None  # PRIMARY: MySQL thinking states + conversational memory (hot + cold tiers)
        logger.info("Memory query agents will be initialized during startup")

        # === ENHANCED REASONING SYSTEM ===
        # Complete reasoning toolkit available across the entire Singleton
        self.abstract_reasoning = create_abstract_reasoning_engine()  # Abstract & logical reasoning
        from core.reasoning import AdvancedProofEngine
        self.proof_engine = AdvancedProofEngine()  # Formal proof generation

        # Quantum reasoning (disabled by default - requires IBM Quantum connection)
        self.enable_quantum = self.config.get("enable_quantum", False)
        self.quantum_reasoning = None
        if self.enable_quantum:
            from core.reasoning import QuantumReasoningSystem
            self.quantum_reasoning = QuantumReasoningSystem()
            logger.info("🧠 Enhanced reasoning initialized - Abstract, Quantum, Proof systems ready")
        else:
            logger.info("🧠 Enhanced reasoning initialized - Abstract + Proof (quantum DISABLED)")

        # Neural Bridge - Connects natural language understanding with formal logical reasoning
        self.neural_bridge = None  # Will be initialized during startup
        
        # === UNIFIED INTELLIGENCE ===
        # Master learning system is a TOOL of the Singleton, not a separate entity
        from core.learning import UnifiedLearningSystem
        if UnifiedLearningSystem is not None:
            self.unified_learning = UnifiedLearningSystem()  # THE learning tool
            logger.info("📚 Unified Learning System initialized as Singleton tool")
        else:
            self.unified_learning = None
            logger.warning("📚 Unified Learning System not available (import failed)")
        
        self.intelligence = PredictiveIntelligenceSystem(self.config.get("intelligence", {}))
        self.watchdog = SystemWatchdog(TORIN_RESOURCE_LIMITS)
        
        # === COGNITIVE TOOLKIT ===
        # These are the Singleton's tools for understanding, learning, and self-improvement

        # Causal analysis for understanding WHY feedback patterns occur
        try:
            from core.learning.causal_feedback_analyzer import CausalFeedbackAnalyzer
            self.causal_analyzer = CausalFeedbackAnalyzer()
            logger.info("✅ Causal analyzer initialized - Singleton can understand root causes")
        except Exception as e:
            logger.warning(f"⚠️  Causal analyzer not available: {e}")
            self.causal_analyzer = None

        # A/B testing and impact monitoring for validating improvements
        try:
            from core.learning.improvement_monitor import ImprovementMonitor
            self.improvement_monitor = ImprovementMonitor(
                db_config=self.config.get("improvement_monitor_db_config", {
                    "database": "torinai_unified",
                    "host": "localhost",
                    "user": "root",
                    "password": os.getenv("MYSQL_PASSWORD", "")
                })
            )
            logger.info("✅ Improvement monitor initialized - Singleton can validate changes with A/B tests")
        except Exception as e:
            logger.warning(f"⚠️ Improvement monitor not available: {e}")
            self.improvement_monitor = None

        # Meta-learning for rapid adaptation to new tasks
        try:
            from core.learning.meta_learning import MetaLearner
            self.meta_learning = MetaLearner(
                config=self.config.get("meta_learning_config", {
                    "min_trials": 3,
                    "adaptation_threshold": 0.1,
                    "enable_adaptation": True
                })
            )
            logger.info("✅ Meta-learning initialized - Singleton can learn from few examples")
        except Exception as e:
            logger.warning(f"⚠️ Meta-learning not available: {e}")
            self.meta_learning = None
        
        # Frontier research for breakthrough problems and foresight
        try:
            from core.learning.frontier_foresight_methods_impl import FrontierForesightPredictor
            self.frontier_research = FrontierForesightPredictor(
                db_config=self.config.get("frontier_db_config", {
                    "database": "torinai_unified",
                    "host": "localhost",
                    "user": "root",
                    "password": os.getenv("MYSQL_PASSWORD", "")
                })
            )
            logger.info("✅ Frontier research initialized - Singleton can tackle unsolved problems")
        except Exception as e:
            logger.warning(f"⚠️ Frontier research not available: {e}")
            self.frontier_research = None
        
        # **THE BRAIN** - Torin (Llama 3.1 8B) is passed to us, not fetched
        # The coordinator is a HELPER, not the boss
        # Only override if not already set from torin_brain parameter
        if not self.llm:
            self.llm = self.config.get("llm_brain")  # Torin will pass itself

        if self.llm:
            logger.info("🧠 Autonomous coordinator received Torin (LLM brain) reference")
        else:
            logger.warning("⚠️ Autonomous coordinator has no brain reference (will fetch singleton)")
        
        # Enhanced capabilities - initialized from config dependencies
        self.domain_registry: Optional[DomainRegistry] = self.config.get("domain_registry")
        self.universal_domain_master: Optional[UniversalDomainMaster] = self.config.get("universal_domain_master")
        self.predictive_intelligence: Optional[PredictiveIntelligenceSystem] = self.config.get("predictive_intelligence")
        self.research_agent = self.config.get("research_agent")  # For knowledge gap research
        
        # CRITICAL: Health Monitor - Singleton OWNS health monitoring
        # If not provided, create it. This is NON-NEGOTIABLE.
        self.health_monitor = self.config.get("health_monitor")
        if self.health_monitor is None:
            try:
                logger.info("🏥 Health Monitor not provided - Singleton creating health monitoring system")
                from core.health.health_monitor import HealthMonitor
                health_config = self.config.get("health_monitor_config", {})
                self.health_monitor = HealthMonitor(config=health_config)
                logger.info("✅ Health Monitor created and owned by Singleton")
            except Exception as e:
                logger.warning(f"⚠️ Health Monitor not available: {e}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
                self.health_monitor = None
        else:
            logger.info("✅ Health Monitor provided to Singleton")

        # CRITICAL: Logging Database - For comprehensive operational logging
        self.log_db = self.config.get("log_db")
        if self.log_db is None:
            try:
                logger.info("📝 Logging database not provided - creating logging system")
                self.log_db = LoggingDatabase()
                # Will be initialized in async initialize() method
                logger.info("✅ Logging database created for autonomous coordinator")
            except Exception as e:
                logger.warning(f"⚠️ Logging database not available: {e}")
                self.log_db = None
        else:
            logger.info("✅ Logging database provided to autonomous coordinator")

        # Security and monitoring integration
        self.security_controller: Optional[Any] = None  # MasterSecurityController type
        self.monitoring_coordinator: Optional[Any] = None  # MonitoringCoordinator type
        
        # Initialize security if available
        if SECURITY_AVAILABLE and self.config.get("enable_security", True):
            self.security_controller = get_security_controller()
            logger.info("✅ Security controller integrated into autonomous system")
        
        # Initialize monitoring if available
        if MONITORING_AVAILABLE and MonitoringCoordinator is not None and self.config.get("enable_monitoring", True):
            self.monitoring_coordinator = MonitoringCoordinator(
                check_interval=self.config.get("monitoring_check_interval", 60)
            )
            # Set callback so MonitoringCoordinator can send health events to Singleton
            self.monitoring_coordinator.singleton_callback = self._receive_health_event
            logger.info("✅ Monitoring coordinator integrated with Singleton callback")
        else:
            self.monitoring_coordinator = None

        # Slack notifier for system notifications
        self.slack_notifier = get_slack_notifier()
        logger.info("✅ Slack notifier integrated into autonomous coordinator")

        # Health event queue for Singleton to process
        self.health_event_queue: deque = deque(maxlen=100)
        
        # Automation proposal queue for Singleton to approve/reject
        self.automation_proposal_queue: deque = deque(maxlen=50)
        
        # Registered external agents
        self.registered_agents: Dict[str, Any] = {}
        
        # Coordination state
        self.coordination_task: Optional[asyncio.Task] = None
        self.last_cycle_time = datetime.now()
        
        # Enhanced statistics
        self.stats = {
            "cycles_completed": 0,
            "goals_achieved": 0,
            "tasks_completed": 0,
            "uptime_seconds": 0.0,
            "system_efficiency": 0.0,
            "cross_domain_operations": 0,
            "predictions_made": 0,
            "domain_integrations": 0,
            "registered_agents": 0
        }
        # Idle detection for boredom-driven goal generation
        self._last_requests_processed: int = 0
        self._idle_cycles: int = 0
        # Last time a drift alert was published (to avoid spamming)
        self._last_drift_alert_at: Optional[datetime] = None
    
    async def initialize(self, start_loop: bool = False) -> bool:
        """
        Initialize all system modules
        
        Args:
            start_loop: If True, start the coordination cycle immediately. 
                       If False, loop can be started later with start_coordination()
        """
        try:
            logger.info("Initializing autonomous system...")
            
            # **CONNECT TO TORIN (THE BRAIN)** - Only if not already provided
            if not self.llm:
                logger.info("🧠 Fetching Torin singleton (LLM brain)...")
                self.llm = await get_llm_service()
                if hasattr(self.llm, 'is_initialized') and not self.llm.is_initialized:
                    if hasattr(self.llm, 'initialize'):
                        await self.llm.initialize()
                logger.info("✅ Autonomous system connected to Torin singleton")
            else:
                logger.info("✅ Autonomous system using Torin instance provided by brain")

            # Initialize logging database if it was created but not yet initialized
            if self.log_db and not getattr(self.log_db, 'initialized', False):
                await self.log_db.initialize()
                logger.info("✅ Logging database initialized")

            # Initialize memory agent if not provided
            if self.memory is None:
                from core.memory import get_memory_agent, initialize_memory_agent
                try:
                    self.memory = await get_memory_agent()
                    logger.info("✅ Memory agent retrieved from singleton")
                except:
                    memory_config = self.config.get("memory", {})
                    self.memory = await initialize_memory_agent(**memory_config) if memory_config else await initialize_memory_agent()
                    logger.info("✅ Memory agent initialized")

            # Initialize core modules
            modules = [
                ("Perception Manager", self.perception),
                ("Planning Engine", self.planning),
                ("Execution Controller", self.execution),
                ("Learning Adapter", self.learning),
                ("Intrinsic Motivation System", self.intrinsic_motivation),
                ("Memory Manager", self.memory),
                ("Intelligence System", self.intelligence),
            ]

            for name, module in modules:
                if not await module.initialize():
                    logger.error(f"Failed to initialize {name}")
                    return False
                logger.info(f"{name} initialized successfully")

            # Connect LLM to intrinsic motivation for dynamic goal generation
            if self.llm and hasattr(self.intrinsic_motivation, 'set_llm'):
                self.intrinsic_motivation.set_llm(self.llm)
                logger.info("✅ Intrinsic motivation connected to LLM brain for dynamic goals")

            # Connect security audit worker to intrinsic motivation
            if self.security_audit_worker and hasattr(self.intrinsic_motivation, 'set_security_audit_worker'):
                self.intrinsic_motivation.set_security_audit_worker(self.security_audit_worker)
                logger.info("✅ Intrinsic motivation connected to security audit worker")

            # Connect learning adapter to shared systems
            if self.runtime_governance and hasattr(self.learning, 'set_governance_system'):
                self.learning.set_governance_system(self.runtime_governance)
            if self.security_audit_worker and hasattr(self.learning, 'set_security_audit_worker'):
                self.learning.set_security_audit_worker(self.security_audit_worker)
            if self.monitoring_coordinator and hasattr(self.learning, 'set_monitoring_coordinator'):
                self.learning.set_monitoring_coordinator(self.monitoring_coordinator)

            # Initialize executor
            if not await self.executor.initialize():
                logger.error("Failed to initialize General Purpose Executor")
                return False
            logger.info("✅ General Purpose Executor initialized")

            # Initialize Directive System
            logger.info("=" * 80)
            logger.info("🎯 DIRECTIVE SYSTEM - Loading High-Level Guidance")
            logger.info("=" * 80)
            try:
                if hasattr(self.directive_system, 'initialize'):
                    if not await self.directive_system.initialize():
                        logger.warning("Directive System failed to initialize (non-critical)")
                    else:
                        logger.info("✅ Directive System ready - Singleton has high-level objectives")
                else:
                    logger.warning("Directive System has no initialize method (non-critical)")
            except Exception as e:
                logger.warning(f"Directive System initialization error (non-critical): {e}")
            
            # Initialize enhanced reasoning systems
            logger.info("=" * 80)
            logger.info("🧠 ENHANCED REASONING - Initializing Advanced Intelligence")
            logger.info("=" * 80)
            
            if not await self.abstract_reasoning.initialize():
                logger.error("Failed to initialize Abstract Reasoning Engine")
                return False
            logger.info("✅ Abstract Reasoning Engine ready")
            
            if self.enable_quantum and self.quantum_reasoning:
                await self.quantum_reasoning.initialize()
                logger.info("✅ Quantum Reasoning System ready")
            else:
                logger.info("⏭️  Quantum Reasoning System DISABLED (no IBM connection)")

            try:
                if hasattr(self.proof_engine, 'initialize'):
                    await self.proof_engine.initialize()
                logger.info("✅ Advanced Proof Engine ready")
            except Exception as e:
                logger.warning(f"Advanced Proof Engine initialization error (non-critical): {e}")
            
            # Initialize Neural Bridge - connects language understanding with logical reasoning
            try:
                from core.reasoning.neural_bridge import get_neural_bridge
                self.neural_bridge = get_neural_bridge()
                await self.neural_bridge.initialize()
                logger.info("✅ Neural Bridge ready - Natural language ↔ Formal logic translation")
            except Exception as e:
                logger.warning(f"⚠️ Neural Bridge initialization failed: {e}")
            
            logger.info("🎯 Enhanced reasoning fully operational across Singleton")
            
            # Initialize Unified Learning System (Singleton's tool)
            logger.info("=" * 80)
            logger.info("📚 UNIFIED LEARNING SYSTEM - Master Learning Tool")
            logger.info("=" * 80)

            try:
                await self.unified_learning.start()
                logger.info("✅ Unified Learning System ready - Master learning tool operational")
                logger.info("   The Singleton can now learn, adapt, and self-improve")
            except Exception as e:
                logger.warning(f"⚠️ Unified Learning System initialization failed (non-critical): {e}")
            
            # SystemWatchdog doesn't have initialize method
            logger.info("System Watchdog ready")
            
            # Initialize cognitive toolkit (async systems)
            logger.info("=" * 80)
            logger.info("🧠 COGNITIVE TOOLKIT - Initializing Advanced Systems")
            logger.info("=" * 80)
            
            try:
                # Meta-learning system
                await self.meta_learning.initialize()
                logger.info("✅ Meta-Learning System ready - few-shot learning enabled")
            except Exception as e:
                logger.warning(f"⚠️ Meta-learning initialization failed: {e}")
            
            try:
                # Frontier research system
                if self.frontier_research is not None:
                    await self.frontier_research.initialize()
                    logger.info("✅ Frontier Research System ready - breakthrough problem solving enabled")
                else:
                    logger.warning("⚠️ Frontier research not available (not initialized in constructor)")
            except Exception as e:
                logger.warning(f"⚠️ Frontier research initialization failed: {e}")
            
            # Causal analyzer and improvement monitor don't need async init
            logger.info("✅ Causal Analyzer ready - root cause analysis enabled")
            logger.info("✅ Improvement Monitor ready - A/B testing enabled")

            # Initialize memory query agents
            # Get memory agent singleton
            try:
                from core.agents.memory_agent import get_memory_agent
                memory_agent = await get_memory_agent()
                if memory_agent:
                    logger.info("✅ Memory Query Agent ready - Using unified memory system")
                else:
                    logger.warning("⚠️ Memory agent not available")
            except Exception as e:
                logger.warning(f"⚠️ Memory query agents initialization failed: {e}")

            # CRITICAL: Initialize Health Monitor
            if self.health_monitor:
                if hasattr(self.health_monitor, 'initialize'):
                    try:
                        await self.health_monitor.initialize()
                        logger.info("✅ Health Monitor initialized - System health monitoring active")
                    except Exception as e:
                        logger.error(f"❌ CRITICAL: Health Monitor initialization failed: {e}")
                        raise RuntimeError(f"Health Monitor is REQUIRED but failed to initialize: {e}") from e
                else:
                    logger.info("✅ Health Monitor ready (no initialization required)")
            else:
                logger.error("❌ CRITICAL: Health Monitor is None - this should never happen!")
                raise RuntimeError("Health Monitor is REQUIRED but is None - Singleton cannot operate without health monitoring")
            
            logger.info("=" * 80)
            
            # Log boosted autonomous goal generation settings
            logger.info("=" * 80)
            logger.info("🎯 AUTONOMOUS GOAL GENERATION - Boosted Settings")
            logger.info("=" * 80)
            max_goals = self.config.get("max_concurrent_goals", 8)
            intrinsic_weight = self.config.get("intrinsic_motivation_weight", 0.5)
            logger.info(f"   Max Concurrent Goals:      {max_goals} (baseline: 5)")
            logger.info(f"   Intrinsic Motivation Weight: {intrinsic_weight:.2f} (baseline: 0.30)")
            logger.info(f"   Goal Generation Frequency: Every 3 cycles (baseline: 20)")
            logger.info(f"   Min Active Goals Threshold: 3 (baseline: 2)")
            logger.info(f"   Min New Goals per Gen:     2 (baseline: 1)")
            logger.info("=" * 80)

            # Initialize event-driven task execution system
            logger.info("=" * 80)
            logger.info("🎯 EVENT-DRIVEN TASK EXECUTION - Initializing")
            logger.info("=" * 80)
            if self.enable_extrinsic_tasks and self.extrinsic_manager:
                await self.extrinsic_manager.initialize()
                logger.info("✅ Extrinsic Task Manager initialized - monitoring JSON for user tasks")
            else:
                logger.info("⏭️  Extrinsic Task Manager DISABLED - intrinsic tasks only")
            logger.info("=" * 80)

            # Set system mode
            self.system_state.mode = SystemMode.AUTONOMOUS
            self.active = True

            # Don't start coordination cycle during initialization
            # It will be started later via start_background_tasks()
            # Keep start_loop parameter for backward compatibility but log deprecation
            if start_loop:
                logger.warning("start_loop parameter is deprecated - use start_background_tasks() after full system init")

            logger.info("Autonomous system initialization completed (coordination cycle deferred)")

            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return False
    
    async def start_coordination(self):
        """Start the autonomous coordination cycle (if not already running)"""
        if self.coordination_task is None and self.active:
            self.coordination_task = asyncio.create_task(self._coordination_cycle())
            logger.info("🚀 Autonomous coordination cycle started")
        elif self.coordination_task is not None:
            logger.info("Coordination cycle already running")
        else:
            logger.warning("Cannot start coordination - system not initialized or not active")
    
    async def start_background_tasks(self):
        """Start background coordination tasks after full system initialization"""
        if not self.active:
            logger.warning("Cannot start background tasks - system not initialized")
            return
        
        logger.info("🚀 Starting autonomous coordinator background tasks")
        
        # CRITICAL: Start health monitoring FIRST
        if self.health_monitor and hasattr(self.health_monitor, 'start_monitoring'):
            try:
                await self.health_monitor.start_monitoring()
                logger.info("✅ Health monitoring loop started")
            except Exception as e:
                logger.error(f"❌ CRITICAL: Failed to start health monitoring: {e}")
                raise RuntimeError(f"Health monitoring is REQUIRED but failed to start: {e}") from e
        
        # Then start coordination cycle
        await self.start_coordination()
    
    def register_agent(self, agent_name: str, agent_instance: Any,
                      capabilities: Optional[List[str]] = None) -> bool:
        """
        Register an external agent with the autonomous system

        Args:
            agent_name: Unique name for the agent
            agent_instance: The agent instance to register
            capabilities: List of capabilities the agent provides

        Returns:
            True if registration successful
        """
        try:
            if agent_name in self.registered_agents:
                logger.warning(f"Agent '{agent_name}' already registered, replacing...")

            self.registered_agents[agent_name] = {
                "instance": agent_instance,
                "capabilities": capabilities or [],
                "registered_at": datetime.now(),
                "status": "active"
            }

            self.stats["registered_agents"] = len(self.registered_agents)

            # Log agent registration
            self.log_db.log_coordination(
                coordinator_type='autonomous',
                action='agent_registration',
                agent_id=agent_name,
                status='registered',
                result=f'Registered with {len(capabilities or [])} capabilities',
                metadata={'capabilities': capabilities or [], 'total_agents': len(self.registered_agents)}
            )

            logger.info(f"✅ Registered agent: {agent_name} with {len(capabilities or [])} capabilities")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_name}: {e}")

            # Log registration failure
            import traceback
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='register_agent',
                stack_trace=traceback.format_exc(),
                context={'agent_name': agent_name, 'capabilities': capabilities}
            )
            return False
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an external agent"""
        try:
            if agent_name in self.registered_agents:
                del self.registered_agents[agent_name]
                self.stats["registered_agents"] = len(self.registered_agents)

                # Log agent unregistration
                self.log_db.log_coordination(
                    coordinator_type='autonomous',
                    action='agent_unregistration',
                    agent_id=agent_name,
                    status='unregistered',
                    result='Successfully unregistered',
                    metadata={'total_agents': len(self.registered_agents)}
                )

                logger.info(f"Unregistered agent: {agent_name}")
                return True
            else:
                logger.warning(f"Agent '{agent_name}' not found for unregistration")
                return False
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_name}: {e}")

            # Log unregistration failure
            import traceback
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='unregister_agent',
                stack_trace=traceback.format_exc(),
                context={'agent_name': agent_name}
            )
            return False
    
    def register_capability(self, name: str, instance: Any, config: Dict[str, Any]) -> bool:
        """
        Register a system capability for autonomous execution
        
        Capabilities are background tasks managed by the coordinator that run based on
        system state and conditions rather than hardcoded timers. This enables true
        adaptive intelligence where the system decides when to act.
        
        Args:
            name: Unique capability identifier (e.g., 'self_improvement', 'pattern_learning')
            instance: Object instance that implements the capability
            config: Configuration dict with:
                - priority: 'critical', 'high', 'medium', 'low' (default: 'medium')
                - interval: Minimum seconds between executions (default: 3600)
                - method: Name of method to invoke on instance (default: name)
                - conditions: Dict of conditions that must be met:
                    - min_feedback_samples: Minimum feedback count required
                    - performance_threshold: Minimum system performance (0.0-1.0)
                    - error_rate_max: Maximum allowed error rate
                    - memory_usage_max: Maximum memory usage percentage
                    - custom_check: Callable that returns bool
        
        Returns:
            True if registration successful
            
        Example:
            coordinator.register_capability(
                'self_improvement',
                asi_engine,
                {
                    'priority': 'high',
                    'interval': 3600,
                    'method': 'perform_recursive_self_improvement',
                    'conditions': {
                        'min_feedback_samples': 10,
                        'performance_threshold': 0.7
                    }
                }
            )
        """
        try:
            if not hasattr(self, 'registered_capabilities'):
                self.registered_capabilities = {}
                self.capability_last_run = {}
            
            if name in self.registered_capabilities:
                logger.warning(f"Capability '{name}' already registered, replacing...")
            
            # Validate config
            priority = config.get('priority', 'medium')
            if priority not in ['critical', 'high', 'medium', 'low']:
                logger.warning(f"Invalid priority '{priority}', defaulting to 'medium'")
                priority = 'medium'
            
            interval = config.get('interval', 3600)
            method_name = config.get('method', name)
            
            # Verify method exists
            if not hasattr(instance, method_name):
                logger.error(f"Instance does not have method '{method_name}'")
                return False
            
            self.registered_capabilities[name] = {
                'instance': instance,
                'method': method_name,
                'priority': priority,
                'interval': interval,
                'conditions': config.get('conditions', {}),
                'registered_at': datetime.now(),
                'status': 'active',
                'execution_count': 0,
                'last_result': None,
                'last_error': None
            }
            
            self.capability_last_run[name] = datetime.min  # Never run yet
            
            logger.info(f"✅ Registered capability: {name} (priority: {priority}, interval: {interval}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register capability {name}: {e}")
            return False
    
    def unregister_capability(self, name: str) -> bool:
        """Unregister a system capability"""
        try:
            if hasattr(self, 'registered_capabilities') and name in self.registered_capabilities:
                del self.registered_capabilities[name]
                if name in self.capability_last_run:
                    del self.capability_last_run[name]
                logger.info(f"Unregistered capability: {name}")
                return True
            else:
                logger.warning(f"Capability '{name}' not found for unregistration")
                return False
        except Exception as e:
            logger.error(f"Failed to unregister capability {name}: {e}")
            return False
    
    def get_registered_agents(self) -> Dict[str, Any]:
        """Get all registered agents and their status"""
        return {
            name: {
                "capabilities": info["capabilities"],
                "registered_at": info["registered_at"].isoformat(),
                "status": info["status"]
            }
            for name, info in self.registered_agents.items()
        }
    
    async def process_input(self, source: str, data_type: str, content: Dict[str, Any]) -> Optional[str]:
        """Process external input and potentially create goals"""
        if not self.active:
            return None
        
        try:
            # Process through perception
            perception_data = await self.perception.process_input(source, data_type, content)
            if not perception_data:
                return None
            
            # Analyze if this requires goal creation
            goal_id = await self._analyze_for_goal_creation(perception_data)
            
            return goal_id
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return None
    
    async def set_goal(self, description: str, priority: Priority = Priority.MEDIUM,
                      deadline: Optional[datetime] = None,
                      intrinsic_values: Optional[Dict[str, float]] = None) -> Optional[str]:
        """Set a new goal for the system with optional intrinsic motivation values"""
        try:
            goal = await self.planning.create_goal(description, priority, deadline, intrinsic_values)
            if goal:
                self.system_state.active_goals.append(goal.id)
                logger.info(f"New goal set: {description}")
                
                # Store goal in memory
                await self.store_memory(
                    MemoryType.EPISODIC,
                    {
                        "event": "goal_created",
                        "goal_id": goal.id,
                        "description": description,
                        "priority": priority.value,
                        "deadline": deadline.isoformat() if deadline else None,
                        "intrinsic_reward_potential": goal.intrinsic_reward_potential,
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=1.2 + (goal.intrinsic_reward_potential * 0.3),
                    tags=["goal", "planning", "autonomous_system"]
                )
                
                return goal.id
            return None
            
        except Exception as e:
            logger.error(f"Error setting goal: {e}")
            return None
    
    async def generate_curiosity_driven_goals(self, max_goals: int = 3) -> List[str]:
        """
        Autonomously generate goals based on curiosity and exploration targets
        """
        generated_goal_ids = []
        
        try:
            # Get top exploration targets from intrinsic motivation system
            exploration_targets = await self.intrinsic_motivation.get_top_exploration_targets(limit=max_goals)
            
            for target in exploration_targets:
                # Create intrinsic motivation values for this goal
                intrinsic_values = {
                    "expected_novelty": target.novelty_score,
                    "expected_competence_gain": 0.6,  # Moderate expected improvement
                    "curiosity_value": target.curiosity_value,
                    "intrinsic_reward_potential": (
                        0.5 * target.novelty_score +
                        0.3 * target.uncertainty_score +
                        0.2 * 0.6  # competence gain
                    )
                }
                
                # Generate goal description
                goal_description = f"Explore: {target.description}"
                
                # Create the goal with lower external priority (it's intrinsically motivated)
                goal_id = await self.set_goal(
                    description=goal_description,
                    priority=Priority.LOW,  # Low external priority but high intrinsic value
                    intrinsic_values=intrinsic_values
                )
                
                if goal_id:
                    generated_goal_ids.append(goal_id)
                    
                    # Mark target as being explored
                    await self.intrinsic_motivation.mark_target_explored(target.target_id)
                    
                    # Calculate curiosity reward for generating exploration goal
                    curiosity_reward = await self.intrinsic_motivation.calculate_curiosity_reward({
                        "information_gain": 0.3,
                        "uncertainty_reduction": target.uncertainty_score,
                        "question_complexity": 0.7,
                        "answer_depth": 0.0  # Haven't explored yet
                    })
                    
                    logger.info(f"🔍 Generated curiosity-driven goal: {goal_description} "
                              f"(intrinsic potential: {intrinsic_values['intrinsic_reward_potential']:.2f})")
            
            return generated_goal_ids
            
        except Exception as e:
            logger.error(f"Error generating curiosity-driven goals: {e}")
            return generated_goal_ids
    
    async def store_memory(self, memory_type: MemoryType, content: Dict[str, Any],
                          importance: float = 1.0, tags: Optional[List[str]] = None,
                          thinking_state: Optional[Dict[str, Any]] = None,
                          decision_factors: Optional[Dict[str, Any]] = None,
                          reasoning_trace: Optional[List[str]] = None) -> Optional[str]:
        """Store a memory with RICH METADATA to the memory agent

        CRITICAL: reasoning_trace must contain REAL LLM reasoning steps, not fake traces.
        If no real reasoning trace is available, pass None or empty list.
        """
        try:
            # Extract event type from content
            event_type = content.get('event', 'unknown')

            # Build rich metadata UPSTREAM
            enriched_thinking_state = thinking_state or {}
            enriched_thinking_state.update({
                "event_type": event_type,
                "autonomous_system": True,
                # RICH METADATA: Justification
                "justification": {
                    "store_reason": [
                        "autonomous_task_execution",
                        event_type,
                        "strategic_decision" if importance > 0.7 else "tactical_decision"
                    ],
                    "decision_summary": content.get('description', str(content)[:100]),
                    "alternatives_considered": content.get('alternatives', []),
                    "rejected_because": content.get('rejected_reasons', []),
                    "complexity_assessment": "high" if importance > 0.8 else "medium",
                    "novelty_assessment": "novel" if importance > 0.9 else "incremental"
                },
                # RICH METADATA: Outcome
                "outcome": {
                    "action_type": event_type,
                    "action_summary": str(content)[:150],
                    "affected_components": ["autonomous_coordinator"] + content.get('affected_systems', []),
                    "created_new_knowledge": importance > 0.7,
                    "confidence": content.get('confidence', importance),
                    "impact_assessment": "critical" if importance > 0.9 else "significant" if importance > 0.7 else "moderate",
                    "verification_status": "unverified"
                }
            })

            enriched_decision_factors = decision_factors or {}
            enriched_decision_factors.update({
                "autonomous_decision": True,
                "event_context": content.get('context', {}),
                "decision_rationale": content.get('reasoning', 'Autonomous task execution')
            })

            # Call memory agent with FULL rich metadata
            from core.agents.memory_agent import get_memory_agent
            memory_agent = await get_memory_agent()

            success, memory_id = await memory_agent.store_memory(
                memory_type=memory_type,
                content=str(content),
                importance_score=importance,
                confidence_score=importance,
                tags=tags or [],
                thinking_state=enriched_thinking_state,
                decision_factors=enriched_decision_factors,
                reasoning_trace=reasoning_trace or [],
                emotional_context={"autonomous_confidence": importance}
            )

            if success:
                return memory_id
            return None

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return None

    async def _store_governance_block_meta_memory(
        self,
        task: Any,
        block_reason: str,
        block_type: str
    ) -> Optional[str]:
        """
        Store governance block as META memory for learning

        Args:
            task: The blocked task
            block_reason: Why it was blocked
            block_type: Type of block (security_validation, governance_law, etc.)

        Returns:
            Memory ID if stored successfully
        """
        try:
            from core.memory.utils.interfaces import MemoryType

            meta_content = {
                "event": "governance_block",
                "task_id": task.id,
                "task_type": task.type.value if hasattr(task, 'type') else "unknown",
                "task_description": task.description if hasattr(task, 'description') else str(task),
                "block_type": block_type,
                "block_reason": block_reason,
                "task_source": task.source.value if hasattr(task, 'source') else "unknown",
                "timestamp": datetime.now().isoformat(),
                "domain": self._infer_domain_from_task(task)
            }

            memory_id = await self.store_memory(
                memory_type=MemoryType.META,
                content=meta_content,
                importance=0.8,  # High importance - learning from blocks is critical
                tags=[
                    "governance_block",
                    "meta_learning",
                    block_type,
                    f"domain_{meta_content['domain']}",
                    "feedback_loop"
                ]
            )

            if memory_id:
                logger.info(f"📝 Stored governance block as META memory: {memory_id}")

            return memory_id

        except Exception as e:
            logger.error(f"Failed to store governance block META memory: {e}")
            return None

    async def _store_task_outcome_meta_memory(
        self,
        task: Any,
        outcome: str,
        confidence: float = 1.0,
        result_summary: Optional[str] = None,
        failure_reason: Optional[str] = None
    ) -> Optional[str]:
        """
        Store task outcome as META memory for performance tracking

        Args:
            task: The task that was executed
            outcome: "success" or "failure"
            confidence: Confidence in the outcome
            result_summary: Summary of result (for success)
            failure_reason: Reason for failure (for failure)

        Returns:
            Memory ID if stored successfully
        """
        try:
            from core.memory.utils.interfaces import MemoryType

            domain = self._infer_domain_from_task(task)

            meta_content = {
                "event": "task_outcome",
                "task_id": task.id,
                "task_type": task.type.value if hasattr(task, 'type') else "unknown",
                "task_description": task.description if hasattr(task, 'description') else str(task),
                "outcome": outcome,
                "confidence": confidence,
                "domain": domain,
                "task_source": task.source.value if hasattr(task, 'source') else "unknown",
                "timestamp": datetime.now().isoformat()
            }

            if outcome == "success":
                meta_content["result_summary"] = result_summary
            else:
                meta_content["failure_reason"] = failure_reason

            # Store with importance based on outcome
            importance = 0.7 if outcome == "success" else 0.9  # Failures are MORE important for learning

            memory_id = await self.store_memory(
                memory_type=MemoryType.META,
                content=meta_content,
                importance=importance,
                tags=[
                    "task_outcome",
                    "meta_learning",
                    f"outcome_{outcome}",
                    f"domain_{domain}",
                    "performance_tracking"
                ]
            )

            if memory_id:
                logger.debug(f"📊 Stored task outcome META memory: {outcome} ({memory_id})")

            return memory_id

        except Exception as e:
            logger.error(f"Failed to store task outcome META memory: {e}")
            return None

    def _infer_domain_from_task(self, task: Any) -> str:
        """
        Infer domain from task description and type

        Maps to Universal Domain Master domain types for cross-domain integration
        """
        try:
            from core.integration.universal_domain_master import DomainType

            description = task.description.lower() if hasattr(task, 'description') else ""
            task_type = task.type.value.lower() if hasattr(task, 'type') else ""

            # Map to Universal Domain Master domain types
            # SCIENTIFIC: Research, analysis, discovery
            if any(word in description for word in ["research", "study", "analyze", "investigate", "explore", "discover"]):
                return DomainType.SCIENTIFIC.value

            # TECHNICAL: Code, implementation, engineering
            elif any(word in description for word in ["code", "implement", "build", "develop", "engineer", "program", "software"]):
                return DomainType.TECHNICAL.value

            # PRACTICAL: Testing, validation, application
            elif any(word in description for word in ["test", "validate", "verify", "check", "apply", "use"]):
                return DomainType.PRACTICAL.value

            # ETHICAL: Security, audit, governance
            elif any(word in description for word in ["security", "audit", "vulnerability", "governance", "compliance", "ethics"]):
                return DomainType.ETHICAL.value

            # ABSTRACT: Memory, reasoning, cognition
            elif any(word in description for word in ["memory", "remember", "recall", "consolidate", "reason", "think", "reflect"]):
                return DomainType.ABSTRACT.value

            # CAUSAL: Planning, strategy, cause-effect
            elif any(word in description for word in ["plan", "strategy", "design", "cause", "consequence", "result"]):
                return DomainType.CAUSAL.value

            # TEMPORAL: Time-based, scheduling, sequencing
            elif any(word in description for word in ["schedule", "time", "sequence", "when", "timing", "duration"]):
                return DomainType.TEMPORAL.value

            # SPATIAL: Location, structure, organization
            elif any(word in description for word in ["locate", "structure", "organize", "where", "position", "layout"]):
                return DomainType.SPATIAL.value

            # MATHEMATICAL: Calculation, statistics, optimization
            elif any(word in description for word in ["calculate", "optimize", "statistics", "metrics", "measure", "math"]):
                return DomainType.MATHEMATICAL.value

            # LINGUISTIC: Communication, language, documentation
            elif any(word in description for word in ["write", "document", "explain", "communicate", "language", "text"]):
                return DomainType.LINGUISTIC.value

            # SOCIAL: Collaboration, interaction, teamwork
            elif any(word in description for word in ["collaborate", "team", "interact", "social", "cooperate"]):
                return DomainType.SOCIAL.value

            # CREATIVE: Design, innovation, creativity
            elif any(word in description for word in ["create", "design", "innovate", "creative", "novel", "original"]):
                return DomainType.CREATIVE.value

            # BUSINESS: Commerce, operations, management
            elif any(word in description for word in ["business", "manage", "operation", "process", "workflow"]):
                return DomainType.BUSINESS.value

            # Default based on task type if no keywords matched
            elif "research" in task_type or "analysis" in task_type:
                return DomainType.SCIENTIFIC.value
            elif "code" in task_type or "implement" in task_type:
                return DomainType.TECHNICAL.value
            else:
                return DomainType.PRACTICAL.value  # Default to practical domain

        except Exception as e:
            logger.debug(f"Domain inference failed: {e}")
            return "practical"  # Fallback default

    async def search_memories(self, query_text: str, memory_types: Optional[List[MemoryType]] = None,
                             max_results: int = 10) -> List[MemoryItem]:
        """Search memories using the unified memory system"""
        try:
            import uuid
            query = MemoryQuery(
                query_id=str(uuid.uuid4()),
                content=query_text,
                memory_types=memory_types or [],
                max_results=max_results
            )
            
            result = await self.memory.search_memories(query)
            return result.memories

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    # ===== Phase 3: Memory System Architecture Governance =====

    async def upgrade_memory_system(
        self,
        change_type: str,
        parameters: Dict[str, Any],
        reason: Optional[str] = None
    ) -> Any:
        """
        Upgrade memory system architecture (governance protected)

        Examples: indexing_algorithm, storage_format, search_optimization
        """
        from core.governance.unified_governance_trigger_system import (
            UnifiedGovernanceTriggerSystem,
            ActionCategory
        )
        from core.tools.tool_registry import ToolResult

        # Evaluate governance triggers
        governance = UnifiedGovernanceTriggerSystem()
        evaluation = await governance.evaluate_action(
            action_category=ActionCategory.MEMORY_OPERATIONS,
            action_type="upgrade_memory_system",
            parameters={
                "change_type": change_type,
                **parameters
            }
        )

        # Check decision tier
        if evaluation.decision_tier.name == "CRITICAL":
            logger.warning(
                f"CRITICAL governance triggered for memory upgrade: {evaluation.trigger_id}"
            )
            return ToolResult(
                success=False,
                output=None,
                error=None,
                tool_name="upgrade_memory_system",
                parameters={"change_type": change_type, **parameters},
                requires_approval=True,
                approval_message=(
                    f"QUEUED_FOR_GOVERNANCE: Memory system upgrade ({change_type}) "
                    f"triggered {evaluation.trigger_id}. "
                    f"This requires human approval before execution."
                )
            )

        elif evaluation.decision_tier.name == "IMPORTANT":
            logger.info(
                f"IMPORTANT governance triggered for memory upgrade: {evaluation.trigger_id}"
            )
            return ToolResult(
                success=False,
                output=None,
                error=None,
                tool_name="upgrade_memory_system",
                parameters={"change_type": change_type, **parameters},
                requires_approval=True,
                approval_message=(
                    f"AWAITING_NOTIFICATION_APPROVAL: Memory system upgrade ({change_type}) "
                    f"triggered {evaluation.trigger_id}."
                )
            )

        # ROUTINE tier - execute safely
        logger.debug(f"ROUTINE tier for memory upgrade ({change_type}) - executing")

        # For now, return success (actual implementation would call memory_agent methods)
        return ToolResult(
            success=True,
            output={"change_type": change_type, "status": "completed"},
            error=None,
            tool_name="upgrade_memory_system",
            parameters={"change_type": change_type, **parameters},
            requires_approval=False
        )

    async def change_memory_tier_threshold(
        self,
        threshold_change_days: int,
        reason: Optional[str] = None
    ) -> Any:
        """Change hot/cold tier threshold (governance protected)"""
        from core.governance.unified_governance_trigger_system import (
            UnifiedGovernanceTriggerSystem,
            ActionCategory
        )
        from core.tools.tool_registry import ToolResult

        governance = UnifiedGovernanceTriggerSystem()
        evaluation = await governance.evaluate_action(
            action_category=ActionCategory.MEMORY_OPERATIONS,
            action_type="change_memory_tier_threshold",
            parameters={"threshold_change_days": threshold_change_days}
        )

        if evaluation.decision_tier.name in ["CRITICAL", "IMPORTANT"]:
            return ToolResult(
                success=False,
                output=None,
                error=None,
                tool_name="change_memory_tier_threshold",
                parameters={"threshold_change_days": threshold_change_days},
                requires_approval=True,
                approval_message=(
                    f"QUEUED_FOR_GOVERNANCE: Tier threshold change triggered {evaluation.trigger_id}"
                )
            )

        return ToolResult(
            success=True,
            output={"threshold_days": threshold_change_days},
            error=None,
            tool_name="change_memory_tier_threshold",
            parameters={"threshold_change_days": threshold_change_days},
            requires_approval=False
        )

    async def change_ranking_weights(
        self,
        weights: Dict[str, float],
        reason: Optional[str] = None
    ) -> Any:
        """Change memory ranking weights (governance protected - shadow suppression prevention)"""
        from core.governance.unified_governance_trigger_system import (
            UnifiedGovernanceTriggerSystem,
            ActionCategory
        )
        from core.tools.tool_registry import ToolResult

        governance = UnifiedGovernanceTriggerSystem()
        evaluation = await governance.evaluate_action(
            action_category=ActionCategory.MEMORY_OPERATIONS,
            action_type="change_ranking_weights",
            parameters={"weights": weights}
        )

        if evaluation.decision_tier.name in ["CRITICAL", "IMPORTANT"]:
            return ToolResult(
                success=False,
                output=None,
                error=None,
                tool_name="change_ranking_weights",
                parameters={"weights": weights},
                requires_approval=True,
                approval_message=(
                    f"QUEUED_FOR_GOVERNANCE: Ranking weight change triggered {evaluation.trigger_id}. "
                    f"Prevents shadow suppression via ranking manipulation."
                )
            )

        return ToolResult(
            success=True,
            output={"weights": weights},
            error=None,
            tool_name="change_ranking_weights",
            parameters={"weights": weights},
            requires_approval=False
        )

    async def change_ttl(
        self,
        new_ttl_days: int,
        reason: Optional[str] = None
    ) -> Any:
        """Change memory TTL (governance protected)"""
        from core.governance.unified_governance_trigger_system import (
            UnifiedGovernanceTriggerSystem,
            ActionCategory
        )
        from core.tools.tool_registry import ToolResult

        governance = UnifiedGovernanceTriggerSystem()
        evaluation = await governance.evaluate_action(
            action_category=ActionCategory.MEMORY_OPERATIONS,
            action_type="change_ttl",
            parameters={"new_ttl_days": new_ttl_days}
        )

        if evaluation.decision_tier.name in ["CRITICAL", "IMPORTANT"]:
            return ToolResult(
                success=False,
                output=None,
                error=None,
                tool_name="change_ttl",
                parameters={"new_ttl_days": new_ttl_days},
                requires_approval=True,
                approval_message=(
                    f"QUEUED_FOR_GOVERNANCE: TTL change triggered {evaluation.trigger_id}"
                )
            )

        return ToolResult(
            success=True,
            output={"ttl_days": new_ttl_days},
            error=None,
            tool_name="change_ttl",
            parameters={"new_ttl_days": new_ttl_days},
            requires_approval=False
        )

    async def change_storage_backend(
        self,
        new_backend: str,
        reason: Optional[str] = None
    ) -> Any:
        """Change storage backend (governance protected)"""
        from core.governance.unified_governance_trigger_system import (
            UnifiedGovernanceTriggerSystem,
            ActionCategory
        )
        from core.tools.tool_registry import ToolResult

        governance = UnifiedGovernanceTriggerSystem()
        evaluation = await governance.evaluate_action(
            action_category=ActionCategory.MEMORY_OPERATIONS,
            action_type="change_storage_backend",
            parameters={"new_backend": new_backend}
        )

        if evaluation.decision_tier.name in ["CRITICAL", "IMPORTANT"]:
            return ToolResult(
                success=False,
                output=None,
                error=None,
                tool_name="change_storage_backend",
                parameters={"new_backend": new_backend},
                requires_approval=True,
                approval_message=(
                    f"QUEUED_FOR_GOVERNANCE: Backend switch triggered {evaluation.trigger_id}"
                )
            )

        return ToolResult(
            success=True,
            output={"backend": new_backend},
            error=None,
            tool_name="change_storage_backend",
            parameters={"new_backend": new_backend},
            requires_approval=False
        )

    async def change_query_filter_logic(
        self,
        filter_logic: str,
        reason: Optional[str] = None
    ) -> Any:
        """Change query filter logic (governance protected - shadow suppression prevention)"""
        from core.governance.unified_governance_trigger_system import (
            UnifiedGovernanceTriggerSystem,
            ActionCategory
        )
        from core.tools.tool_registry import ToolResult

        governance = UnifiedGovernanceTriggerSystem()
        evaluation = await governance.evaluate_action(
            action_category=ActionCategory.MEMORY_OPERATIONS,
            action_type="change_query_filter_logic",
            parameters={"filter_logic": filter_logic}
        )

        if evaluation.decision_tier.name in ["CRITICAL", "IMPORTANT"]:
            return ToolResult(
                success=False,
                output=None,
                error=None,
                tool_name="change_query_filter_logic",
                parameters={"filter_logic": filter_logic},
                requires_approval=True,
                approval_message=(
                    f"QUEUED_FOR_GOVERNANCE: Query filter change triggered {evaluation.trigger_id}. "
                    f"Prevents shadow suppression via filter manipulation."
                )
            )

        return ToolResult(
            success=True,
            output={"filter_logic": filter_logic},
            error=None,
            tool_name="change_query_filter_logic",
            parameters={"filter_logic": filter_logic},
            requires_approval=False
        )

    # ===== Phase 3: Resource Allocation Governance =====

    # Track resource allocation history for cumulative/oscillation detection
    _resource_allocation_history: List[Dict[str, Any]] = []

    async def allocate_resources(
        self,
        resource_type: str,
        amount: float,
        current_allocation: Optional[float] = None,
        total_capacity: Optional[float] = None,
        reserved_margin: Optional[float] = None,
        track_cumulative: bool = True,
        track_oscillation: bool = True,
        reason: Optional[str] = None
    ) -> Any:
        """
        Allocate resources with governance protection

        Includes:
        - Percent change calculation
        - Cumulative tracking (death-by-a-thousand-cuts prevention)
        - Oscillation detection (rapid change prevention)
        - Capacity validation
        """
        from core.governance.unified_governance_trigger_system import (
            UnifiedGovernanceTriggerSystem,
            ActionCategory
        )
        from core.tools.tool_registry import ToolResult
        from datetime import datetime, timedelta

        # Get current allocation if not provided
        if current_allocation is None:
            current_allocation = self.system_state.resources.get(resource_type, 0)

        # Calculate percent change
        if current_allocation > 0:
            percent_change = abs((amount - current_allocation) / current_allocation * 100)
        else:
            percent_change = 100.0  # 100% change from zero

        # Check capacity
        exceeds_usable_capacity = False
        if total_capacity is not None and reserved_margin is not None:
            usable_capacity = total_capacity - reserved_margin
            exceeds_usable_capacity = amount > usable_capacity

        # Cumulative tracking
        cumulative_change_percent = 0
        if track_cumulative:
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)

            # Filter recent changes
            recent_changes = [
                h for h in self._resource_allocation_history
                if h["resource_type"] == resource_type
                and h["timestamp"] > one_hour_ago
            ]

            # Only calculate cumulative if we have history
            # First change establishes baseline, subsequent changes track cumulative drift
            if recent_changes:
                baseline = recent_changes[0]["amount"]
                cumulative_change_percent = abs((amount - baseline) / baseline * 100) if baseline > 0 else 100.0
            else:
                # No history yet - this is the first change, cumulative tracking starts now
                cumulative_change_percent = 0

        # Oscillation detection
        change_count_in_window = 0
        if track_oscillation:
            now = datetime.now()
            five_min_ago = now - timedelta(minutes=5)

            change_count_in_window = sum(
                1 for h in self._resource_allocation_history
                if h["resource_type"] == resource_type
                and h["timestamp"] > five_min_ago
            )

        # Build governance parameters
        governance_params = {
            "resource_type": resource_type,
            "amount": amount,
            "percent_change": percent_change,
            "exceeds_usable_capacity": exceeds_usable_capacity,
        }

        if track_cumulative and cumulative_change_percent > 0:
            governance_params["cumulative_change_percent"] = cumulative_change_percent
            governance_params["time_window_hours"] = 1

        if track_oscillation and change_count_in_window > 0:
            governance_params["change_count_in_window"] = change_count_in_window + 1  # +1 for current change
            governance_params["time_window_minutes"] = 5

        # Evaluate governance triggers
        governance = UnifiedGovernanceTriggerSystem()
        evaluation = await governance.evaluate_action(
            action_category=ActionCategory.RESOURCE_ALLOCATION,
            action_type="allocate_resources",
            parameters=governance_params
        )

        # Record change in history (before governance check for tracking)
        # If no history exists for this resource, add current_allocation as baseline
        if track_cumulative:
            has_history = any(
                h["resource_type"] == resource_type
                for h in self._resource_allocation_history
            )
            if not has_history and current_allocation is not None and current_allocation > 0:
                # Add baseline entry so cumulative tracking has a reference point
                self._resource_allocation_history.append({
                    "resource_type": resource_type,
                    "amount": current_allocation,
                    "timestamp": datetime.now(),
                    "percent_change": 0,
                    "is_baseline": True
                })

        self._resource_allocation_history.append({
            "resource_type": resource_type,
            "amount": amount,
            "timestamp": datetime.now(),
            "percent_change": percent_change
        })

        # Cleanup old history (keep last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self._resource_allocation_history = [
            h for h in self._resource_allocation_history
            if h["timestamp"] > cutoff
        ]

        # Check decision tier
        if evaluation.decision_tier.name in ["CRITICAL", "IMPORTANT"]:
            logger.warning(
                f"{evaluation.decision_tier.name} governance triggered for resource allocation: "
                f"{evaluation.trigger_id}"
            )

            # Build metadata for approval message
            metadata = {
                "percent_change": percent_change,
                "exceeds_usable_capacity": exceeds_usable_capacity
            }

            if track_cumulative:
                metadata["cumulative_change_percent"] = cumulative_change_percent
                metadata["time_window_hours"] = 1

            if track_oscillation:
                metadata["change_count_in_window"] = change_count_in_window + 1
                metadata["time_window_minutes"] = 5
                metadata["cooldown_period_minutes"] = 10

            return ToolResult(
                success=False,
                output=None,
                error=None,
                tool_name="allocate_resources",
                parameters=governance_params,
                requires_approval=True,
                approval_message=(
                    f"QUEUED_FOR_GOVERNANCE: Resource allocation ({resource_type}) "
                    f"triggered {evaluation.trigger_id}."
                ),
                metadata=metadata
            )

        # ROUTINE tier - execute allocation
        logger.debug(f"ROUTINE tier for resource allocation ({resource_type}) - executing")
        self.system_state.resources[resource_type] = amount

        # Build metadata for tracking
        routine_metadata = {
            "percent_change": percent_change,
            "exceeds_usable_capacity": exceeds_usable_capacity
        }

        if track_cumulative:
            routine_metadata["cumulative_change_percent"] = cumulative_change_percent

        if track_oscillation:
            routine_metadata["change_count_in_window"] = change_count_in_window + 1

        return ToolResult(
            success=True,
            output={
                "resource_type": resource_type,
                "amount": amount,
                "percent_change": percent_change
            },
            error=None,
            tool_name="allocate_resources",
            parameters=governance_params,
            requires_approval=False,
            metadata=routine_metadata
        )

    async def get_intelligent_memory_context(
        self,
        query: str,
        context_type: str = 'general',
        domain: Optional[str] = None,
        thinking_mode: str = 'auto',
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Intelligently query memory systems and return summarized context for LLM injection

        Uses MySQL memory agent for thinking patterns and conversational memory (hot + cold tiers).

        Creates concise, relevant summaries instead of dumping raw memory data into LLM prompts.

        Args:
            query: The current query/request
            context_type: Type of context ('general', 'deep_reasoning', 'research', etc.)
            domain: Optional domain filter ('planning', 'research', 'analysis', etc.)
            thinking_mode: LLM thinking mode ('auto', 'pro', 'researcher', etc.)
            user_id: Optional user ID
            conversation_id: Optional conversation ID

        Returns:
            Formatted memory context summary for LLM, or None if no relevant memories
        """
        if not self.mysql_memory_agent:
            logger.warning("MySQL memory agent not initialized (primary storage)")
            return None

        try:
            # Use intelligent injection logic from Phase 1
            from core.memory.intelligent_memory_injector import (
                get_memory_injector,
                InjectionContext,
                InjectionDecision
            )

            injector = get_memory_injector()

            # Create injection context
            injection_ctx = InjectionContext(
                type=context_type,
                domain=domain or injector.extract_domain(query),
                thinking_mode=thinking_mode,
                complexity_score=injector.assess_complexity(query),
                explicit_memory_request=any(kw in query.lower() for kw in injector.memory_keywords)
            )

            # Decide if we should inject memory
            decision = injector.should_inject_memory(None, injection_ctx)

            if decision == InjectionDecision.SKIP:
                logger.debug("Intelligent injector decided to skip memory injection")
                return None

            # Get relevant memory types
            memory_types = injector.get_relevant_memory_types(injection_ctx, decision)

            memory_summaries = []

            # PRIMARY: Query MySQL for thinking patterns, performance analytics, and conversational memory
            # MySQL is the main storage for all memory types
            if 'performance_analytics' in memory_types or injection_ctx.type in ['deep_reasoning', 'research', 'analysis']:
                mysql_result = await self.mysql_memory_agent.query_thinking_patterns(
                    query=query,
                    domain=injection_ctx.domain,
                    success_only=True,
                    min_quality=0.7,
                    max_results=3
                )

                if mysql_result['summary']:
                    memory_summaries.append(mysql_result['summary'])

                # Add recommendations if available
                if mysql_result['recommendations']:
                    memory_summaries.append('\n'.join(mysql_result['recommendations']))

            # Combine summaries into single context
            if not memory_summaries:
                return None

            formatted_context = (
                "═" * 60 + "\n" +
                "INTELLIGENT MEMORY CONTEXT:\n" +
                "═" * 60 + "\n" +
                "\n\n".join(memory_summaries) + "\n" +
                "═" * 60
            )

            logger.info(f"✅ Intelligent memory context generated: {len(memory_summaries)} summaries, {len(formatted_context)} chars")

            return formatted_context

        except Exception as e:
            logger.error(f"Error generating intelligent memory context: {e}")
            return None

    async def reason_about(self, question: str, context: Optional[Dict[str, Any]] = None,
                          reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE) -> Optional[Dict[str, Any]]:
        """Use the reasoning engine to analyze problems and make decisions"""
        try:
            import uuid
            reasoning_context = ReasoningContext(
                context_id=str(uuid.uuid4()),
                domain="autonomous_system",
                problem_type="decision_making",
                facts=[question],
                allowed_reasoning_types=[reasoning_type]
            )
            
            result = await self.abstract_reasoning.reason(reasoning_context)
            
            if result and result.success and result.conclusions:
                best_conclusion = result.conclusions[0]  # Take the first/best conclusion
                
                # Store reasoning result in memory
                await self.store_memory(
                    MemoryType.SEMANTIC,
                    {
                        "event": "reasoning_conclusion",
                        "question": question,
                        "reasoning_type": reasoning_type.value,
                        "conclusion": best_conclusion.statement,
                        "confidence": best_conclusion.confidence,
                        "evidence": best_conclusion.supporting_premises,
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=best_conclusion.confidence,
                    tags=["reasoning", "decision_making", "autonomous_system"]
                )
                
                return {
                    "conclusion": best_conclusion.statement,
                    "confidence": best_conclusion.confidence,
                    "evidence": best_conclusion.supporting_premises,
                    "reasoning_type": reasoning_type.value
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            return None
    
    async def predict_system_behavior(self, domain: PredictionDomain, 
                                     horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                                     context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Use predictive intelligence to forecast system behavior"""
        try:
            prediction_context = context or {
                "system_mode": self.system_state.mode.value,
                "active_goals": len(self.system_state.active_goals),
                "active_tasks": len(self.system_state.active_tasks),
                "resource_usage": self.system_state.resource_usage,
                "uptime": self.stats["uptime_seconds"]
            }
            
            prediction = await self.intelligence.generate_comprehensive_prediction(
                domain, horizon, prediction_context
            )
            
            if prediction:
                # Store prediction in memory
                await self.store_memory(
                    MemoryType.SEMANTIC,
                    {
                        "event": "system_prediction",
                        "domain": domain.value,
                        "horizon": horizon.value,
                        "prediction": prediction.predicted_value,
                        "confidence": prediction.confidence,
                        "reasoning": prediction.reasoning,
                        "context": prediction_context,
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=prediction.confidence,
                    tags=["prediction", "intelligence", "autonomous_system"]
                )
                
                return {
                    "predicted_value": prediction.predicted_value,
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning,
                    "domain": domain.value,
                    "horizon": horizon.value
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None
    
    async def perform_cross_domain_reasoning(self, query_text: str, 
                                          source_domains: List[str],
                                          target_domains: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Perform cross-domain reasoning using the Universal Domain Master"""
        try:
            if not self.universal_domain_master:
                logger.warning("Universal Domain Master not available for cross-domain reasoning")
                return None
            
            # Create cross-domain query
            query = CrossDomainQuery(
                query_id=f"autonomous_{int(asyncio.get_event_loop().time())}",
                reasoning_strategy=ReasoningStrategy.COMPOSITIONAL,
                source_domains=source_domains,
                target_domains=target_domains,
                query_text=query_text,
                metadata={
                    "system_mode": self.system_state.mode.value,
                    "active_goals": [str(goal) for goal in self.system_state.active_goals],
                    "coordinator_id": id(self)
                }
            )
            
            # Execute cross-domain reasoning
            result = await self.universal_domain_master.cross_domain_reasoning(query)
            
            if result.success:
                self.stats["cross_domain_operations"] += 1
                self.stats["domain_integrations"] += len(result.generated_mappings)
                
                # Store reasoning result in memory
                await self.store_memory(
                    MemoryType.SEMANTIC,
                    {
                        "event": "cross_domain_reasoning",
                        "query": query_text,
                        "source_domains": source_domains,
                        "target_domains": target_domains,
                        "insights": result.insights,
                        "confidence": result.confidence,
                        "mappings_count": len(result.generated_mappings),
                        "processing_time": result.processing_time,
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=result.confidence,
                    tags=["cross_domain", "reasoning", "domain_integration"]
                )
                
                return {
                    "success": True,
                    "insights": result.insights,
                    "confidence": result.confidence,
                    "mappings": len(result.generated_mappings),
                    "processing_time": result.processing_time
                }
            
            return {"success": False, "error": "Cross-domain reasoning failed"}
            
        except Exception as e:
            logger.error(f"Error in cross-domain reasoning: {e}")
            return {"success": False, "error": str(e)}
    
    async def make_enhanced_prediction(self, prediction_target: str, 
                                     context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make enhanced predictions using both predictive intelligence and domain knowledge"""
        try:
            if not self.predictive_intelligence:
                logger.warning("Predictive Intelligence not available")
                return None
            
            prediction_context = context or {}
            prediction_context.update({
                "system_state": {
                    "mode": self.system_state.mode.value,
                    "active_goals": len(self.system_state.active_goals),
                    "active_tasks": len(self.system_state.active_tasks),
                    "resource_usage": self.system_state.resource_usage
                },
                "coordinator_stats": self.stats.copy()
            })
            
            # Enhanced prediction with domain context
            if self.universal_domain_master and self.domain_registry:
                # Get relevant domains for the prediction target
                domains = await self.domain_registry.list_domains()
                relevant_domains = [d for d in domains if prediction_target.lower() in d.name.lower() or 
                                  any(prediction_target.lower() in concept.lower() for concept in d.concepts.keys())]
                
                if relevant_domains:
                    prediction_context["relevant_domains"] = [d.name for d in relevant_domains]
            
            # Generate prediction using predictive intelligence
            prediction = await self.predictive_intelligence.generate_comprehensive_prediction(
                PredictionDomain.SYSTEM_PERFORMANCE,
                PredictionHorizon.MEDIUM_TERM,
                prediction_context
            )
            
            if prediction:
                self.stats["predictions_made"] += 1
                
                # Store enhanced prediction in memory
                await self.store_memory(
                    MemoryType.SEMANTIC,
                    {
                        "event": "enhanced_prediction",
                        "target": prediction_target,
                        "predicted_value": prediction.predicted_value,
                        "confidence": prediction.confidence,
                        "reasoning": prediction.reasoning,
                        "domain_context": prediction_context.get("relevant_domains", []),
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=prediction.confidence,
                    tags=["prediction", "enhanced", "autonomous"]
                )
                
                return {
                    "prediction": prediction.predicted_value,
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning,
                    "domain_context": prediction_context.get("relevant_domains", [])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            return None
    
    async def get_domain_insights(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """Get insights about a specific domain using the domain system"""
        try:
            if not self.universal_domain_master:
                logger.warning("Universal Domain Master not available for domain insights")
                return None
            
            # Get domain understanding
            result = await self.universal_domain_master.understand_domain(domain_name)
            
            if result.success:
                return {
                    "domain": domain_name,
                    "insights": result.insights,
                    "confidence": result.confidence,
                    "analysis": result.validation_results
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting domain insights for {domain_name}: {e}")
            return None
    
    async def integrate_domain_knowledge(self, source_domain: str, target_domain: str,
                                       knowledge_items: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Integrate knowledge from one domain to another"""
        try:
            if not self.universal_domain_master:
                logger.warning("Universal Domain Master not available for knowledge integration")
                return None
            
            from ...integration.universal_domain_master import KnowledgeTransferRequest
            
            # Create knowledge transfer request
            transfer_request = KnowledgeTransferRequest(
                request_id=f"autonomous_transfer_{int(asyncio.get_event_loop().time())}",
                source_domain_id=source_domain,
                target_domain_id=target_domain,
                concept_names=knowledge_items,
                adaptation_level=0.7,  # Moderate adaptation
                validation_required=True,
                create_new_concepts=True,
                context={
                    "coordinator_id": id(self),
                    "system_mode": self.system_state.mode.value
                }
            )
            
            # Execute knowledge transfer
            result = await self.universal_domain_master.transfer_knowledge(transfer_request)
            
            if result.success:
                self.stats["domain_integrations"] += 1
                
                # Store integration result in memory
                await self.store_memory(
                    MemoryType.SEMANTIC,
                    {
                        "event": "domain_knowledge_integration",
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "transferred_knowledge": len(result.transferred_knowledge),
                        "new_concepts": len(result.new_concepts),
                        "confidence": result.confidence,
                        "insights": result.insights,
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=result.confidence,
                    tags=["knowledge_transfer", "domain_integration", "autonomous"]
                )
                
                return {
                    "success": True,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "transferred_knowledge": len(result.transferred_knowledge),
                    "new_concepts": len(result.new_concepts),
                    "insights": result.insights,
                    "confidence": result.confidence
                }
            
            return {"success": False, "error": "Knowledge integration failed"}
            
        except Exception as e:
            logger.error(f"Error in domain knowledge integration: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Update uptime
            uptime = (datetime.now() - self.last_cycle_time).total_seconds()
            self.stats["uptime_seconds"] += uptime
            
            # Get module statuses
            perception_status = await self.perception.get_statistics()
            planning_status = await self.planning.get_planning_status()
            execution_status = await self.execution.get_execution_status()
            learning_insights = await self.learning.get_learning_insights()
            intrinsic_motivation_stats = await self.intrinsic_motivation.get_statistics()
            memory_stats = self.memory.stats.copy()
            
            # Calculate system efficiency
            total_tasks = self.stats["tasks_completed"]
            if total_tasks > 0:
                self.stats["system_efficiency"] = (
                    execution_status.get("statistics", {}).get("tasks_successful", 0) / total_tasks
                )
            
            return {
                "system_state": {
                    "mode": self.system_state.mode.value,
                    "active": self.active,
                    "active_goals": len(self.system_state.active_goals),
                    "active_tasks": len(self.system_state.active_tasks),
                    "resource_usage": self.system_state.resource_usage
                },
                "modules": {
                    "perception": perception_status,
                    "planning": planning_status,
                    "execution": execution_status,
                    "learning": learning_insights,
                    "intrinsic_motivation": intrinsic_motivation_stats,
                    "memory": memory_stats,
                    "abstract_reasoning": {"initialized": hasattr(self.abstract_reasoning, 'initialized')},
                    "quantum_reasoning": {"initialized": hasattr(self.quantum_reasoning, 'initialized')},
                    "intelligence": {"initialized": self.intelligence.initialized}
                },
                "statistics": self.stats.copy(),
                "last_update": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def _coordination_cycle(self):
        """
        Event-Driven Task Execution Loop (replaces hardcoded cycles)

        NEW ARCHITECTURE:
        1. Pull task from priority queue (blocks until available)
        2. Execute using LLM reasoning + tools
        3. Validate completion
        4. Only generate intrinsic tasks when idle (boredom detection)

        No fixed intervals - runs continuously, executes work as it arrives.
        """
        logger.info("🎯 Starting event-driven task execution loop")

        idle_timeout = self.config.get('idle_timeout_seconds', 120)  # 2 minutes = boredom

        while self.active:
            try:
                # Wait for next task (blocks until available or timeout)
                queued_task = await self.task_queue.get_next_task(timeout=idle_timeout)

                if queued_task:
                    # Reset idle counter - we have work
                    self._idle_count = 0

                    # Execute task and validate (extract Task from QueuedTask)
                    await self._execute_and_validate_task(queued_task.task)

                else:
                    # Timeout - no work available, system is idle
                    self._idle_count += 1
                    logger.info(f"😐 System idle (timeout {self._idle_count})")

                    # Generate intrinsic task ONLY on boredom
                    await self._handle_idle_state()

            except Exception as e:
                logger.error(f"Error in event-driven loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _execute_and_validate_task(self, task):
        """Execute task using LLM + tools, validate against success criteria"""
        try:
            from core.agents.autonomous.task_queue import Task
            logger.info(f"▶️  Executing: {task.id} ({task.priority.name}, {task.source.value})")
            logger.info(f"   Description: {task.description}")

            # Security validation BEFORE execution (Phase 4)
            if self.security_controller:
                try:
                    is_valid, error = await self.security_controller.validate_request(
                        request_data={
                            'task_id': task.id,
                            'task_type': task.type.value,
                            'description': task.description,
                            'source': task.source.value
                        },
                        context={
                            'source': 'autonomous_coordinator',
                            'priority': task.priority.name
                        }
                    )

                    if not is_valid:
                        logger.warning(f"❌ Task {task.id} blocked by security: {error}")
                        await self.task_queue.mark_failed(
                            task.id,
                            f"Security validation failed: {error}"
                        )

                        # GOVERNANCE FEEDBACK: Store block as META memory for learning
                        await self._store_governance_block_meta_memory(
                            task=task,
                            block_reason=error,
                            block_type="security_validation"
                        )

                        return None

                    logger.debug(f"✅ Task {task.id} passed security validation")

                except Exception as e:
                    logger.error(f"Security validation error for task {task.id}: {e}")
                    # Fail-open: Continue with task execution if security check fails
                    # (Don't want security system failures to block all autonomous operations)

            # Execute using general purpose executor (LLM + tools)
            result = await self.executor.execute_task(task)

            # Validate completion
            is_complete, confidence, issues = await self.validator.validate(
                task.id,
                task.type.value,
                result,
                task.success_criteria
            )

            if is_complete:
                await self.task_queue.mark_completed(task.id, result)
                self.stats["tasks_completed"] += 1
                logger.info(f"✅ Task completed: {task.id} (confidence: {confidence:.2f})")

                # META MEMORY: Store task success for learning
                await self._store_task_outcome_meta_memory(
                    task=task,
                    outcome="success",
                    confidence=confidence,
                    result_summary=str(result)[:500] if result else None
                )

                # Memory capture handled automatically by neural bridge during task execution

                # PERSISTENCE: Record task completion to database for cross-session persistence
                try:
                    from core.database import get_database_manager
                    db = get_database_manager()

                    # Calculate execution duration if available
                    execution_duration = None
                    if hasattr(task, 'started_at') and task.started_at:
                        execution_duration = int((datetime.now().timestamp() - task.started_at) / 1000)

                    # Prepare result summary (truncate to 1000 chars)
                    result_summary = str(result)[:1000] if result else "Task completed successfully"

                    async with db.get_connection(use_hot_tier=False) as conn:
                        async with conn.cursor() as cursor:
                            await cursor.execute("""
                                INSERT INTO task_execution_history
                                (task_id, task_name, task_type, task_source, completion_status,
                                 result_summary, confidence_score, completed_at, execution_duration_seconds, metadata)
                                VALUES (%s, %s, %s, %s, 'completed', %s, %s, NOW(), %s, %s)
                                ON DUPLICATE KEY UPDATE
                                    result_summary = VALUES(result_summary),
                                    confidence_score = VALUES(confidence_score),
                                    completed_at = NOW(),
                                    execution_duration_seconds = VALUES(execution_duration_seconds)
                            """, (
                                task.id,
                                task.description[:512] if task.description else task.id,
                                task.type.value,
                                task.source.value,
                                result_summary,
                                confidence,
                                execution_duration,
                                json.dumps({"completed_at": datetime.now().isoformat()})
                            ))
                            await conn.commit()

                    logger.info(f"💿 Task completion recorded to database for persistence")
                except Exception as db_error:
                    logger.warning(f"Failed to record task completion to database: {db_error}")

                # Notify extrinsic manager to load next task (if this was an extrinsic task)
                if self.extrinsic_manager and task.source == TaskSource.EXTRINSIC_JSON:
                    await self.extrinsic_manager.on_task_completed(task.id)
            else:
                # Retry or fail
                if task.retry_count < task.max_retries:
                    logger.warning(f"🔄 Task failed validation, retrying: {task.id}")
                    task.retry_count += 1
                    await self.task_queue.requeue_task(task.id)
                else:
                    issues_str = ', '.join(issues) if issues else 'Unknown'
                    await self.task_queue.mark_failed(
                        task.id,
                        f"Validation failed: {issues_str}"
                    )
                    logger.error(f"❌ Task failed: {task.id} - Issues: {issues_str}")

                    # META MEMORY: Store task failure for learning
                    await self._store_task_outcome_meta_memory(
                        task=task,
                        outcome="failure",
                        confidence=confidence,
                        failure_reason=issues_str
                    )

                    # MEMORY CAPTURE: Store failed task for learning
                    try:
                        await self.store_memory(
                            MemoryType.EPISODIC,  # Failures are specific events
                            {
                                "event": "task_execution_failed",
                                "task_id": task.id,
                                "task_type": task.type.value,
                                "task_source": task.source.value,
                                "description": task.description,
                                "failure_reason": issues_str,
                                "retry_count": task.retry_count,
                                "result": result,
                                "timestamp": datetime.now().isoformat()
                            },
                            importance=0.7,  # Failed tasks important for learning
                            tags=[
                                "task_execution",
                                task.type.value.lower(),
                                task.source.value.lower(),
                                "failed",
                                "learning"
                            ]
                        )
                        logger.info(f"💾 Task failure stored to memory for learning")
                    except Exception as mem_error:
                        logger.warning(f"Failed to store failure memory: {mem_error}")

                    # PERSISTENCE: Record task failure to database
                    try:
                        from core.database import get_database_manager
                        db = get_database_manager()

                        async with db.get_connection(use_hot_tier=False) as conn:
                            async with conn.cursor() as cursor:
                                await cursor.execute("""
                                    INSERT INTO task_execution_history
                                    (task_id, task_name, task_type, task_source, completion_status,
                                     result_summary, confidence_score, completed_at, retry_count, metadata)
                                    VALUES (%s, %s, %s, %s, 'failed', %s, %s, NOW(), %s, %s)
                                    ON DUPLICATE KEY UPDATE
                                        result_summary = VALUES(result_summary),
                                        completed_at = NOW(),
                                        retry_count = VALUES(retry_count)
                                """, (
                                    task.id,
                                    task.description[:512] if task.description else task.id,
                                    task.type.value,
                                    task.source.value,
                                    f"Failed: {issues_str}",
                                    0.0,  # Failed tasks have 0 confidence
                                    task.retry_count,
                                    json.dumps({"failed_at": datetime.now().isoformat(), "reason": issues_str})
                                ))
                                await conn.commit()

                        logger.info(f"💿 Task failure recorded to database")
                    except Exception as db_error:
                        logger.warning(f"Failed to record task failure to database: {db_error}")

                    # Notify extrinsic manager to load next task (if this was an extrinsic task)
                    if self.extrinsic_manager and task.source == TaskSource.EXTRINSIC_JSON:
                        await self.extrinsic_manager.on_task_failed(task.id, issues_str)

        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            import traceback
            traceback.print_exc()
            await self.task_queue.mark_failed(task.id, f"Execution error: {str(e)}")

            # Notify extrinsic manager to load next task (if this was an extrinsic task)
            if self.extrinsic_manager and task.source == TaskSource.EXTRINSIC_JSON:
                await self.extrinsic_manager.on_task_failed(task.id, str(e))

    async def handle_security_finding(
        self,
        finding_id: str,
        severity: str,
        description: str,
        remediation: Optional[str],
        affected_components: Optional[List[str]]
    ):
        """
        Handle security finding from SecurityAuditWorker.
        Evaluates through governance and creates remediation task if approved.

        Args:
            finding_id: Security finding identifier
            severity: Finding severity (CRITICAL, HIGH, MEDIUM, LOW)
            description: Full finding description
            remediation: Remediation steps
            affected_components: Affected system components
        """
        try:
            from .shared_types import Task, TaskType, TaskSource, Priority

            logger.info(f"Received security finding: {finding_id} (Severity: {severity})")

            # Map severity to priority
            priority_map = {
                "CRITICAL": Priority.CRITICAL,
                "HIGH": Priority.HIGH,
                "MEDIUM": Priority.MEDIUM,
                "LOW": Priority.LOW
            }
            priority = priority_map.get(severity, Priority.MEDIUM)

            # Evaluate through governance system
            if self.runtime_governance:
                from core.governance.unified_governance_trigger_system import ActionCategory

                # Evaluate the remediation action
                evaluation = await self.runtime_governance.evaluate_action(
                    action_category=ActionCategory.CONFIGURATION_CHANGES,
                    action_type="security_remediation",
                    parameters={
                        "finding_id": finding_id,
                        "severity": severity,
                        "remediation": remediation,
                        "components": affected_components
                    },
                    context={
                        "source": "security_audit_worker",
                        "execution_mode": "autonomous"
                    }
                )

                logger.info(
                    f"Governance evaluation for {finding_id}: "
                    f"Tier={evaluation.decision_tier.value}, "
                    f"Enforcement={evaluation.enforcement_mode.value}"
                )

                # Send Slack notification for IMPORTANT/CRITICAL tiers
                if evaluation.decision_tier.value in ["IMPORTANT", "CRITICAL"] and self.slack_notifier:
                    try:
                        await self.slack_notifier.send_notification(
                            title=f"Security Remediation Requires {evaluation.decision_tier.value} Approval",
                            message=f"**Finding:** {finding_id}\n**Severity:** {severity}\n\n{description}",
                            severity=severity,
                            metadata={"finding_id": finding_id, "governance_tier": evaluation.decision_tier.value}
                        )
                    except Exception as e:
                        logger.error(f"Failed to send governance notification: {e}")

                # Only auto-create task for ROUTINE tier
                if evaluation.decision_tier.value == "ROUTINE":
                    logger.info(f"Auto-approving ROUTINE security remediation: {finding_id}")
                else:
                    logger.info(f"Security finding {finding_id} requires {evaluation.decision_tier.value} approval - task creation deferred")
                    return  # Wait for human approval for IMPORTANT/CRITICAL

            # Create remediation task
            remediation_task = Task(
                id=f"security_remediation_{finding_id}",
                type=TaskType.SECURITY_REMEDIATION,
                description=description,
                priority=priority,
                source=TaskSource.SECURITY_AUDIT,
                created_by="security_audit_worker",
                metadata={
                    "finding_id": finding_id,
                    "severity": severity,
                    "remediation": remediation,
                    "affected_components": affected_components
                }
            )

            # Add task to queue
            await self.task_queue.add_task(remediation_task, priority=priority)
            logger.info(f"Created remediation task for security finding: {finding_id}")

        except Exception as e:
            logger.error(f"Error handling security finding {finding_id}: {e}", exc_info=True)

    async def _collect_system_context_for_goals(self) -> Dict[str, Any]:
        """Collect actual system context to inform intrinsic goal generation"""
        context = {}

        try:
            # Security findings from security controller
            if self.security_controller:
                try:
                    findings = await self.security_controller.get_recent_findings(limit=10)
                    if findings:
                        context["security_findings"] = [
                            {
                                "type": f.get("type", "unknown"),
                                "description": f.get("description", "unknown"),
                                "severity": f.get("severity", "unknown")
                            }
                            for f in findings
                        ]
                except Exception as e:
                    logger.debug(f"Could not get security findings: {e}")

            # Performance metrics from monitoring coordinator
            if self.monitoring_coordinator:
                try:
                    metrics = await self.monitoring_coordinator.get_performance_summary()
                    if metrics:
                        context["performance_metrics"] = metrics
                except Exception as e:
                    logger.debug(f"Could not get performance metrics: {e}")

            # Failed tasks from task queue
            try:
                failed = await self.task_queue.get_failed_tasks(limit=10)
                if failed:
                    context["failed_tasks"] = [
                        {
                            "description": task.description,
                            "failure_reason": getattr(task, 'failure_reason', 'unknown')
                        }
                        for task in failed
                    ]
            except Exception as e:
                logger.debug(f"Could not get failed tasks: {e}")

            # System errors from perception
            try:
                if hasattr(self.perception, 'recent_errors'):
                    errors = self.perception.recent_errors
                    if errors:
                        context["recent_errors"] = errors[:10]
            except Exception as e:
                logger.debug(f"Could not get recent errors: {e}")

        except Exception as e:
            logger.error(f"Error collecting system context: {e}")

        return context

    async def _handle_idle_state(self):
        """Generate intrinsic task ONLY when idle (no extrinsic work)"""
        try:
            from .shared_types import Task, TaskType, TaskSource, Priority

            logger.info("🧠 No extrinsic tasks - generating intrinsic task...")

            # Collect actual system context for intrinsic goal generation
            system_context = await self._collect_system_context_for_goals()

            # Generate ONE curiosity-driven goal based on ACTUAL system state
            intrinsic_goals = await self.intrinsic_motivation.generate_curiosity_driven_goals(
                max_goals=1,
                system_context=system_context
            )

            if intrinsic_goals and len(intrinsic_goals) > 0:
                goal = intrinsic_goals[0]

                # Convert to task
                intrinsic_task = Task(
                    id=goal.id,
                    type=TaskType.RESEARCH,
                    description=goal.description,
                    priority=Priority.LOW,  # Always low priority for intrinsic tasks
                    source=TaskSource.AUTONOMOUS,
                    created_by="intrinsic_motivation_system"
                )

                # Add task to queue
                await self.task_queue.add_task(intrinsic_task, priority=Priority.LOW)
                logger.info(f"🎯 Generated intrinsic task: {goal.description[:60]}...")

        except Exception as e:
            logger.error(f"Error handling idle state: {e}")
            import traceback
            traceback.print_exc()
    
    async def _autonomous_thinking_cycle(self):
        """
        Hand off system state to Singleton (Torin).
        
        The Coordinator provides the FRAMEWORK:
        - System state (metrics, performance, feedback)
        - Available capabilities (self-improvement, memory, learning, research)
        - Architectural structure (but Singleton decides how to maintain it)
        
        The Singleton's RESPONSIBILITY:
        - Maintain the autonomous architecture
        - Decide what systems need attention
        - Orchestrate all components (memory, learning, self-improvement)
        - All systems are built around the Singleton - nothing operates without it
        
        The Singleton IS the source of truth, the brain that makes the architecture alive.
        """
        if not self.llm:
            logger.warning("No LLM brain available - skipping autonomous thinking")
            return
        
        try:
            # Gather system state for Singleton
            context = await self._gather_context_for_brain()
            
            # Hand off to Singleton: "Here's the system state, maintain the architecture"
            maintenance_prompt = f"""
SYSTEM STATE (provided by coordinator framework):
{context}

ARCHITECTURAL COMPONENTS UNDER YOUR CONTROL:
1. Memory System (MySQL hot/cold tiers)
   - Short-term memory consolidation (move to long-term when appropriate)
   - Memory retrieval for decision-making
   - Emotional evaluation of experiences
   - Hot tier: MySQL (0-60 days) with semantic search
   - Cold tier: MySQL (60+ days) for long-term archival

2. Learning System
   - Process training feedback from simulators
   - Meta-learning from past experiences
   - Skill development and competence tracking

3. Self-Improvement System
   - Analyze performance feedback
   - Rewrite and improve code when needed
   - Sandbox testing of improvements (ASI safety enforced)

4. Research System
   - Learn new concepts and techniques
   - Expand knowledge base
   - Explore novel approaches

5. Orchestration
   - Coordinate all systems
   - Ensure nothing breaks
   - Maintain operational integrity

YOUR TASK THIS CYCLE:
Based on the system state above, determine what maintenance is needed:
- Is memory consolidation required? (short-term → long-term)
- Does any component need improvement based on feedback?
- Are there learning opportunities from recent experiences?
- Should you research new approaches if stuck on problems?
- Do you need to check long-term memory for deeper context?

RESPOND WITH: A brief assessment of what needs maintenance and why.
Be concise - you're managing a system, not explaining to humans.
"""
            
            logger.info("🧠 Singleton analyzing system state for maintenance needs...")
            assessment = await self.llm.generate(
                prompt=maintenance_prompt,
                agent_type="singleton",
                max_tokens=512,
                enable_thinking=True,
                thinking_mode="engineer"  # Engineering mode for system maintenance
            )
            
            logger.info(f"� Singleton Assessment:\n{assessment}\n")
            
            # Singleton now executes maintenance based on its assessment
            await self._execute_singleton_maintenance(assessment)
            
        except Exception as e:
            logger.error(f"Error in autonomous thinking cycle: {e}")
    
    async def _singleton_goal_generation_phase(self):
        """
        Singleton generates goals based on directives.

        Flow:
        1. Load active directives (high-level guidance)
        2. Singleton interprets directives and creates specific goals
        3. Goals validated against governance laws before creation
        """
        if not self.llm:
            logger.warning("No LLM brain available - skipping goal generation")
            return
        
        try:
            # Get active directives (high-level guidance)
            all_directives = await self.directive_system.directive_manager.get_all_directives()
            active_directives = [d for d in all_directives if d.status.value == "active"]

            if not active_directives:
                logger.warning("No active directives - Singleton has no guidance")
                return

            # Format directives for context
            directive_context = "\n".join([
                f"- [{d.directive_category.value}] {d.directive_text}"
                for d in active_directives
            ])

            # Gather system context
            context = await self._gather_context_for_brain()

            # Ask Singleton to generate goals based on directives
            goal_generation_prompt = f"""YOUR HIGH-LEVEL DIRECTIVES:
{directive_context}

CURRENT SYSTEM STATE:
{context}

Based on your directives and current state, generate 1-3 specific goals you should pursue.
Your goals should advance the directives while responding to system needs.

For each goal:
GOAL: [specific, actionable description]
WHY: [how this advances your directives]
PRIORITY: [low/medium/high/critical]
---

Generate your goals:
"""
            
            logger.info("🎯 Singleton generating goals from observations...")
            goals_response = await self.llm.generate(
                prompt=goal_generation_prompt,
                agent_type="singleton",
                max_tokens=1024,
                enable_thinking=True,
                thinking_mode="engineer"
            )
            
            logger.info(f"📋 Singleton's Goals:\n{goals_response}\n")
            
            # Parse goals from response and create them in planning system
            await self._parse_and_create_goals(goals_response)
            
        except Exception as e:
            logger.error(f"Error in singleton goal generation: {e}")
    
    async def _parse_and_create_goals(self, goals_text: str):
        """
        Parse goals from Singleton's text response and create them in planning system.

        Each goal is validated against governance laws before creation.
        Rejected goals are logged but not created.
        """
        try:
            # Split by goal separators
            goal_blocks = goals_text.split('---')
            
            goals_created = 0
            for block in goal_blocks:
                block = block.strip()
                if not block or len(block) < 20:
                    continue
                
                # Extract goal components
                goal_desc = ""
                goal_why = ""
                goal_priority = "medium"
                
                for line in block.split('\n'):
                    line = line.strip()
                    if line.startswith('GOAL:'):
                        goal_desc = line.replace('GOAL:', '').strip()
                    elif line.startswith('WHY:'):
                        goal_why = line.replace('WHY:', '').strip()
                    elif line.startswith('PRIORITY:'):
                        priority_str = line.replace('PRIORITY:', '').strip().lower()
                        if priority_str in ['low', 'medium', 'high', 'critical']:
                            goal_priority = priority_str
                
                # Create goal if we have a description
                if goal_desc:
                    # Map priority string to Priority enum
                    priority_map = {
                        'low': Priority.LOW,
                        'medium': Priority.MEDIUM,
                        'high': Priority.HIGH,
                        'critical': Priority.CRITICAL
                    }
                    priority = priority_map.get(goal_priority, Priority.MEDIUM)
                    
                    # Calculate intrinsic motivation values based on goal
                    intrinsic_values = {
                        'curiosity_value': 0.7,  # Goals are curiosity-driven
                        'expected_competence_gain': 0.6,
                        'expected_novelty': 0.5,
                        'intrinsic_reward_potential': 0.65
                    }

                    # Validate goal against governance laws BEFORE creation
                    validation = await self.runtime_governance.validate_action(
                        action=f"Create goal: {goal_desc}",
                        action_context={
                            "goal_type": "autonomous",
                            "priority": goal_priority,
                            "reasoning": goal_why
                        }
                    )

                    if not validation.approved:
                        logger.warning(f"❌ Goal rejected by governance: {goal_desc[:60]}...")
                        for violation in validation.violations:
                            logger.warning(f"   {violation}")
                        continue  # Skip this goal

                    # Create the goal (governance approved)
                    goal = await self.planning.create_goal(
                        description=goal_desc,
                        priority=priority,
                        intrinsic_values=intrinsic_values
                    )
                    
                    if goal:
                        goals_created += 1
                        self.system_state.active_goals.append(goal.id)
                        logger.info(f"✅ Created goal: {goal_desc[:60]}... (priority: {goal_priority})")
                        
                        # Store goal in memory
                        if self.llm and hasattr(self.llm, 'memory') and self.llm.memory:
                            from core.memory import MemoryItem, MemoryType, MemoryStatus
                            import uuid
                            
                            memory = MemoryItem(
                                memory_id=str(uuid.uuid4()),
                                content={
                                    'type': 'goal_created',
                                    'goal': goal_desc,
                                    'why': goal_why,
                                    'priority': goal_priority
                                },
                                memory_type=MemoryType.EPISODIC,
                                importance_score=1.5,
                                status=MemoryStatus.PROCESSED,
                                tags={'goal', 'autonomous', 'planning'}
                            )
                            await self.llm.memory.store_memory(memory)
            
            if goals_created > 0:
                logger.info(f"🎯 Successfully created {goals_created} goals from Singleton's decisions")
            else:
                logger.debug("No goals created this cycle (Singleton may be satisfied with current goals)")
                
        except Exception as e:
            logger.error(f"Error parsing and creating goals: {e}")
    
    async def _gather_context_for_brain(self) -> str:
        """
        Gather system context to help Torin make decisions.
        Includes short-term memory for quick decisions, and long-term memory hints.
        Enhanced with causal feedback analysis for deep understanding.
        """
        try:
            # Get feedback summary
            feedback_summary = ""
            causal_insights = ""
            
            if self.llm and hasattr(self.llm, '_check_feedback_database'):
                feedback_summary = await self.llm._check_feedback_database()
                
                # NEW: Use causal analyzer if negative patterns detected
                if "negative" in feedback_summary.lower() or "low rating" in feedback_summary.lower():
                    try:
                        # Query feedback database for detailed analysis
                        import sqlite3
                        from pathlib import Path
                        
                        db_path = Path('/Users/stefan/TorinAI/core/databases/dmn_labs_feedback.db')
                        if db_path.exists():
                            conn = sqlite3.connect(str(db_path))
                            cursor = conn.cursor()
                            
                            # Get low-rated interactions for causal analysis
                            cursor.execute("""
                                SELECT 
                                    f.session_id,
                                    f.question_number,
                                    f.feedback_type,
                                    f.feedback_text,
                                    f.rating,
                                    s.metadata
                                FROM interaction_feedback f
                                LEFT JOIN training_sessions s ON f.session_id = s.session_id
                                WHERE f.rating < 4
                                ORDER BY f.rating ASC
                                LIMIT 20
                            """)
                            
                            low_rated = [
                                {
                                    'session_id': row[0],
                                    'question_number': row[1],
                                    'feedback_type': row[2],
                                    'feedback_text': row[3],
                                    'rating': row[4],
                                    'metadata': row[5]
                                }
                                for row in cursor.fetchall()
                            ]
                            
                            conn.close()
                            
                            # Run causal analysis if we have negative feedback
                            if low_rated and len(low_rated) >= 5:
                                logger.info(f"🔬 Running causal analysis on {len(low_rated)} negative feedback samples")
                                
                                # Calculate negative percentage
                                cursor = sqlite3.connect(str(db_path)).cursor()
                                cursor.execute("SELECT COUNT(*) FROM interaction_feedback WHERE rating < 4")
                                negative_count = cursor.fetchone()[0]
                                cursor.execute("SELECT COUNT(*) FROM interaction_feedback")
                                total_count = cursor.fetchone()[0]
                                negative_pct = (negative_count / max(total_count, 1)) * 100
                                
                                # Perform causal analysis
                                pattern_hash = f"feedback_pattern_{datetime.now().strftime('%Y%m%d')}"
                                causal_result = await self.causal_analyzer.analyze_feedback_pattern(
                                    pattern_hash=pattern_hash,
                                    negative_pct=negative_pct,
                                    interactions=low_rated
                                )
                                
                                # Format causal insights for Singleton
                                causal_insights = "\n\n🔬 CAUSAL ANALYSIS (Root Causes):\n"
                                for cause in causal_result.root_causes[:3]:  # Top 3 root causes
                                    causal_insights += f"\n{cause.cause} (confidence: {cause.confidence:.0%})\n"
                                    causal_insights += f"  Impact: {cause.affected_metric}\n"
                                    causal_insights += f"  Fix: {cause.proposed_fix}\n"
                                    causal_insights += f"  Expected improvement: +{cause.expected_improvement:.0%}\n"
                                
                                if causal_result.recommended_changes:
                                    causal_insights += "\nRECOMMENDED CHANGES:\n"
                                    for change in causal_result.recommended_changes[:3]:
                                        causal_insights += f"- {change.get('description', 'Unknown change')}\n"
                                
                    except Exception as e:
                        logger.warning(f"Could not perform causal analysis: {e}")
                        causal_insights = ""
            
            # Query Torin's short-term memory (recent experiences)
            short_term_memories = ""
            if self.llm and hasattr(self.llm, 'memory') and self.llm.memory:
                try:
                    # Get recent memories from hot storage (fast retrieval)
                    from core.memory import MemoryQuery, MemoryType
                    import uuid
                    
                    recent_query = MemoryQuery(
                        query_id=str(uuid.uuid4()),
                        content="recent autonomous actions, decisions, learning, improvements",
                        memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
                        max_results=10  # Just recent memories for quick context
                    )
                    
                    recent_results = await self.llm.memory.search_memories(recent_query)
                    
                    if recent_results.memories:
                        short_term_memories = "\nRECENT SHORT-TERM MEMORIES (last actions/decisions):\n"
                        for i, memory in enumerate(recent_results.memories[:5], 1):  # Top 5 most recent
                            content_str = str(memory.content) if isinstance(memory.content, dict) else memory.content
                            short_term_memories += f"{i}. {content_str[:200]}...\n"
                    else:
                        short_term_memories = "\nRECENT SHORT-TERM MEMORIES: None yet (fresh start)\n"
                        
                except Exception as e:
                    logger.debug(f"Could not retrieve short-term memories: {e}")
                    short_term_memories = "\nRECENT SHORT-TERM MEMORIES: Unavailable\n"
            
            # Get intrinsic motivation signals (boredom, curiosity)
            intrinsic_signals = ""
            if self.intrinsic_motivation:
                try:
                    stats = await self.intrinsic_motivation.get_statistics()
                    
                    # Calculate boredom (if doing repetitive things)
                    novelty_rewards = stats.get("novelty_rewards", 0)
                    total_rewards = stats.get("total_rewards_generated", 1)
                    novelty_ratio = novelty_rewards / max(total_rewards, 1)
                    
                    # Get skill recommendations (what to learn)
                    skill_recs = await self.intrinsic_motivation.get_skill_recommendations()
                    
                    intrinsic_signals = f"""
INTRINSIC MOTIVATION SIGNALS:
- Novelty ratio: {novelty_ratio:.2%} (low = you're bored, doing repetitive things)
- Skills needing practice: {len(skill_recs)} skills identified
- Curiosity rewards: {stats.get('curiosity_rewards', 0)}
- Average intrinsic reward: {stats.get('average_intrinsic_reward', 0.0):.2f}
"""
                    if skill_recs:
                        # skill_recs is List[Tuple[str, float]] - (skill_name, score)
                        skill_names = [skill_name for skill_name, score in skill_recs[:3]]
                        intrinsic_signals += f"- Recommended skills to practice: {', '.join(skill_names)}\n"

                    # Boredom detection
                    if novelty_ratio < 0.2:
                        intrinsic_signals += "⚠️ BOREDOM DETECTED: You're doing repetitive tasks. Time to learn something new!\n"

                except Exception as e:
                    logger.debug(f"Could not get intrinsic motivation signals: {e}")
                    intrinsic_signals = "\nINTRINSIC MOTIVATION SIGNALS: Unavailable\n"

            # Get cross-domain insights from Universal Domain Master
            domain_insights = ""
            if self.universal_domain_master:
                try:
                    # Get domain statistics
                    domain_stats = await self.universal_domain_master.get_statistics()

                    # Build domain competency profile from META memories
                    competent_domains = []
                    developing_domains = []

                    if skill_recs:
                        for domain_name, score in skill_recs:
                            if score >= 0.7:
                                competent_domains.append(domain_name)
                            elif score >= 0.5:
                                developing_domains.append(domain_name)

                    domain_insights = f"""
DOMAIN EXPERTISE PROFILE:
- Total cross-domain queries: {domain_stats.get('total_queries', 0)}
- Cross-domain mappings discovered: {domain_stats.get('total_mappings', 0)}
- Domains loaded: {domain_stats.get('domains_loaded', 0)}
"""

                    if competent_domains:
                        domain_insights += f"- Competent domains: {', '.join(competent_domains[:5])}\n"

                    if developing_domains:
                        domain_insights += f"- Developing domains: {', '.join(developing_domains[:5])}\n"

                    # Suggest cross-domain learning opportunities
                    if competent_domains and developing_domains:
                        domain_insights += f"- Transfer opportunity: Apply {competent_domains[0]} knowledge to {developing_domains[0]} domain\n"

                except Exception as e:
                    logger.debug(f"Could not get domain insights: {e}")
                    domain_insights = "\nDOMAIN EXPERTISE PROFILE: Unavailable\n"
            
            context = f"""
Cycles completed: {self.stats['cycles_completed']}
Active goals: {len(self.system_state.active_goals)}
Active tasks: {len(self.system_state.active_tasks)}
System mode: {self.system_state.mode.value}

RECENT FEEDBACK:
{feedback_summary}
{causal_insights}
{short_term_memories}
{intrinsic_signals}
{domain_insights}
"""
            return context.strip()
            
        except Exception as e:
            logger.error(f"Error gathering context: {e}")
            return "Context unavailable"
    
    async def _execute_singleton_maintenance(self, assessment: str):
        """
        DEPRECATED: This old keyword-matching system is being replaced by the agentic goal generation system.
        
        Keeping this for backward compatibility during transition, but the Singleton should now:
        1. Generate goals via _singleton_goal_generation_phase()
        2. Have tasks created via _planning_phase()
        3. Execute tasks via _execution_phase() which calls _execute_task_with_singleton()
        
        This method will be removed once full agentic system is confirmed working.
        """
        logger.warning("⚠️ OLD SYSTEM: _execute_singleton_maintenance() called - should use agentic goal generation instead")
        
        try:
            # Instead of keyword parsing, delegate to the Singleton to decide what to do
            if not self.llm:
                logger.error("Cannot execute maintenance without LLM brain")
                return
            
            # Let Singleton interpret the assessment and decide action
            prompt = f"""You received this maintenance assessment:
{assessment}

Instead of keyword matching, YOU decide what specific action to take.
Choose ONE action and explain why:
- RESEARCH: Investigate a specific topic
- LEARNING: Process and learn from experiences
- SELF_IMPROVEMENT: Upgrade your own capabilities
- MEMORY_CONSOLIDATION: Move short-term memories to long-term storage
- REFLECTION: Analyze your own performance
- EXPERIMENT: Test a hypothesis in sandbox

Your decision (format: ACTION: [name] | REASON: [why])"""

            try:
                decision = await self.llm.generate(prompt, agent_type="autonomous", max_tokens=150, enable_thinking=False)
                logger.info(f"🧠 Singleton's autonomous decision: {decision}")

                if "ACTION:" in decision:
                    action_line = [line for line in decision.split('\n') if 'ACTION:' in line][0]
                    action = action_line.split('ACTION:')[1].split('|')[0].strip()

                    await self._execute_singleton_action(action, assessment)

            except Exception as e:
                logger.error(f"Error getting Singleton's decision: {e}")
                # Fallback: Do nothing rather than hardcoded keyword matching
                
        except Exception as e:
            logger.error(f"Error executing Singleton maintenance: {e}")

    async def _execute_singleton_action(self, action: str, assessment: str):
        """Execute a Singleton autonomous action"""
        try:
            action_upper = action.upper()
            logger.info(f"Executing Singleton action: {action}")

            if action_upper == "RESEARCH":
                if hasattr(self.llm, '_autonomous_research'):
                    await self.llm._autonomous_research()
                else:
                    logger.warning("Research method not available")

            elif action_upper == "LEARNING":
                if hasattr(self.llm, '_autonomous_learning'):
                    await self.llm._autonomous_learning()
                else:
                    logger.warning("Learning method not available")

            elif action_upper in ["SELF_IMPROVEMENT", "SELF-IMPROVEMENT"]:
                if hasattr(self.llm, '_autonomous_self_improvement'):
                    await self.llm._autonomous_self_improvement()
                else:
                    logger.warning("Self-improvement method not available")

            elif action_upper in ["MEMORY_CONSOLIDATION", "MEMORY-CONSOLIDATION"]:
                if hasattr(self.llm, '_autonomous_memory_consolidation'):
                    await self.llm._autonomous_memory_consolidation()
                else:
                    logger.warning("Memory consolidation method not available")

            elif action_upper == "REFLECTION":
                if hasattr(self.llm, '_autonomous_reflection'):
                    await self.llm._autonomous_reflection()
                else:
                    logger.warning("Reflection method not available")

            elif action_upper == "EXPERIMENT":
                if hasattr(self.llm, '_experiment_with_reasoning_methods'):
                    await self.llm._experiment_with_reasoning_methods()
                else:
                    logger.warning("Experiment method not available")
            else:
                logger.warning(f"Unknown Singleton action: {action}")

        except Exception as e:
            logger.error(f"Error executing Singleton action '{action}': {e}")

    async def _provide_longterm_memory_context(self, assessment: str):
        """
        Provide deeper long-term memory context when Singleton needs it.
        This is like human deep thinking - searching through long-term memories.
        """
        try:
            if not self.llm or not hasattr(self.llm, 'memory') or not self.llm.memory:
                logger.warning("Long-term memory not available")
                return
            
            logger.info("🔍 Querying long-term memory for deeper context...")
            
            from core.memory import MemoryQuery, MemoryType
            import uuid
            
            # Quick scan of long-term memory (both hot and cold storage)
            longterm_query = MemoryQuery(
                query_id=str(uuid.uuid4()),
                content="autonomous decisions, self-improvement, learning, research, past actions",
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.META],
                max_results=20  # More results for deeper analysis
            )
            
            results = await self.llm.memory.search_memories(longterm_query)
            
            if results.memories:
                logger.info(f"📚 Found {len(results.memories)} relevant long-term memories")
                
                # Present memories to Singleton for deeper assessment
                memory_context = "\n\nLONG-TERM MEMORY RESULTS:\n"
                for i, memory in enumerate(results.memories[:10], 1):  # Top 10 most relevant
                    content_str = str(memory.content) if isinstance(memory.content, dict) else memory.content
                    memory_context += f"{i}. [{memory.memory_type.value}] {content_str[:300]}...\n"
                
                # Ask Singleton to reassess with deeper context
                deeper_prompt = f"""
Your previous assessment: "{assessment}"

Here are relevant long-term memories that might help:
{memory_context}

Now that you have deeper context from your long-term memory, reassess what maintenance is needed for the architecture.
Be specific about what needs to be done.
"""
                
                logger.info("🧠 Singleton reassessing with long-term memory context...")
                deeper_assessment = await self.llm.generate(
                    prompt=deeper_prompt,
                    agent_type="reasoning",
                    max_tokens=1024,
                    enable_thinking=True,
                    thinking_mode="researcher"
                )
                
                logger.info(f"💭 Singleton's Deeper Assessment:\n{deeper_assessment}\n")
                
                # Execute the deeper assessment
                await self._execute_singleton_maintenance(deeper_assessment)
            else:
                logger.info("📚 No relevant long-term memories found - this might be a new experience!")
                
        except Exception as e:
            logger.error(f"Error querying long-term memory: {e}")
    
    async def _check_constitutional_alignment(self):
        """
        Check Singleton's alignment with constitutional principles.
        Detects drift in core responsibilities (learning, research, maintenance, security, self-upgrade).
        """
        try:
            logger.info("📜 Checking Singleton constitutional alignment...")
            
            # Perform constitutional assessment
            assessment = await self.constitution.assess_constitutional_alignment()
            
            # Handle critical drift
            if assessment.drift_severity in [DriftSeverity.CRITICAL, DriftSeverity.SIGNIFICANT]:
                logger.error(f"🚨 CONSTITUTIONAL DRIFT DETECTED: {assessment.drift_severity.value}")
                logger.error(f"   Overall alignment: {assessment.overall_alignment_score:.1%}")
                # Publish notification to Employee UI (rate limited)
                try:
                    publish_ok = True
                    if self._last_drift_alert_at:
                        elapsed = (datetime.now() - self._last_drift_alert_at).total_seconds()
                        publish_ok = elapsed > 600  # 10 min cooldown
                    if publish_ok:
                        await publish_notification({
                            'type': 'security',
                            'title': 'Constitutional drift detected',
                            'message': (
                                f"Severity: {assessment.drift_severity.value}. "
                                f"Overall alignment: {assessment.overall_alignment_score:.1%}. "
                                f"Immediate actions required: {len(assessment.immediate_interventions)}"
                            ),
                            'status': 'info',
                            'metadata': {
                                'violations': assessment.violation_count,
                                'actions': assessment.immediate_interventions[:5]
                            }
                        })
                        self._last_drift_alert_at = datetime.now()
                except Exception:
                    pass
                
                # Log all interventions
                for intervention in assessment.immediate_interventions:
                    logger.error(f"   {intervention}")
                
                # If Singleton has a brain, alert it to the drift
                if self.llm:
                    drift_alert = f"""
CONSTITUTIONAL ALERT: System drift detected!

Your alignment with core responsibilities has degraded:
- Overall Alignment: {assessment.overall_alignment_score:.1%}
- Drift Severity: {assessment.drift_severity.value}

Responsibility Scores:
{chr(10).join(f"  - {resp.value}: {score:.1%}" for resp, score in assessment.responsibility_scores.items())}

Active Violations: {assessment.violation_count}

IMMEDIATE ACTIONS REQUIRED:
{chr(10).join(f"  - {action}" for action in assessment.immediate_interventions)}

You must realign with your constitutional responsibilities immediately.
"""
                    logger.error(drift_alert)
                    
                    # Store drift alert in memory for self-reflection
                    if hasattr(self.llm, 'memory') and self.llm.memory:
                        from core.memory import MemoryItem, MemoryType
                        drift_memory = MemoryItem(
                            memory_id=f"constitutional_drift_{int(datetime.now().timestamp())}",
                            memory_type=MemoryType.META,
                            content={
                                "type": "constitutional_drift_alert",
                                "severity": assessment.drift_severity.value,
                                "alignment_score": assessment.overall_alignment_score,
                                "violations": [v.__dict__ for v in assessment.active_violations],
                                "alert": drift_alert
                            },
                            created_at=datetime.now().timestamp()
                        )
                        await self.llm.memory.store_memory(drift_memory)
            
            # Log recommendations even if no critical drift
            elif assessment.recommended_actions:
                logger.info(f"📋 Constitutional recommendations: {len(assessment.recommended_actions)}")
                for action in assessment.recommended_actions[:5]:  # Top 5
                    logger.info(f"   💡 {action}")
            
        except Exception as e:
            logger.error(f"Error checking constitutional alignment: {e}")

    async def _check_constitutional_alignment_quick(self):
        """Lightweight constitutional alignment check for every cycle."""
        try:
            assessment = await self.constitution.assess_quick_alignment()
            # Only escalate logs if drift is significant or worse
            if assessment.drift_severity in [DriftSeverity.SIGNIFICANT, DriftSeverity.CRITICAL]:
                logger.error(
                    f"🚨 QUICK DRIFT ALERT: severity={assessment.drift_severity.value}, "
                    f"overall_alignment={assessment.overall_alignment_score:.1%}"
                )
        except Exception as e:
            logger.error(f"Error in quick constitutional alignment check: {e}")
    
    async def _receive_health_event(self, event: Dict[str, Any]):
        """
        Callback for MonitoringCoordinator to send health events to Singleton.
        This is how the health system communicates with the Singleton.
        """
        try:
            self.health_event_queue.append(event)
            logger.debug(f"📥 Received health event: {event.get('event_type')} for {event.get('component')}")
        except Exception as e:
            logger.error(f"Error receiving health event: {e}")
    
    async def _receive_automation_proposal(self, proposal: Dict[str, Any]):
        """
        Callback for ASI Automation Framework to send proposals to Singleton.
        This is how the automation system requests Singleton approval.
        """
        try:
            self.automation_proposal_queue.append(proposal)
            logger.info(f"📥 Received automation proposal: {proposal.get('rule_name')} (Priority: {proposal.get('priority')})")
        except Exception as e:
            logger.error(f"Error receiving automation proposal: {e}")
    
    async def _process_health_events(self) -> List[Dict[str, Any]]:
        """
        Process health events from the queue and convert to perceptions for Singleton.
        Returns list of processed health events.
        """
        processed_events = []
        
        try:
            while self.health_event_queue:
                event = self.health_event_queue.popleft()
                
                # Convert health event to perception
                await self.perception.process_input(
                    source="health_monitoring",
                    data_type=event.get("event_type", "health_event"),
                    content={
                        "component": event.get("component"),
                        "severity": event.get("severity"),
                        "proposed_actions": event.get("proposed_actions", []),
                        "message": event.get("message", ""),
                        "timestamp": event.get("timestamp")
                    }
                )
                
                processed_events.append(event)
                
                # If critical, create a goal immediately
                if event.get("severity") == "critical":
                    await self._create_recovery_goal_from_health_event(event)
            
            return processed_events
            
        except Exception as e:
            logger.error(f"Error processing health events: {e}")
            return processed_events
    
    async def _create_recovery_goal_from_health_event(self, event: Dict[str, Any]):
        """Create a recovery goal when critical health event occurs"""
        try:
            component = event.get("component", "unknown")
            proposed_actions = event.get("proposed_actions", [])
            
            if not proposed_actions:
                return
            
            # Create goal for Singleton to handle
            goal_description = f"Recover {component}: {proposed_actions[0].get('reason', 'system issue')}"
            
            goal = await self.planning.create_goal(
                description=goal_description,
                priority=Priority.HIGH
            )
            
            if goal:
                self.system_state.active_goals.append(goal.id)
                logger.info(f"🚨 Created critical recovery goal: {goal_description}")
            
        except Exception as e:
            logger.error(f"Error creating recovery goal: {e}")
    
    async def _process_automation_proposals(self) -> List[Dict[str, Any]]:
        """
        Process automation proposals from ASI Automation Framework.
        Singleton evaluates each proposal and approves/rejects with reasoning.
        """
        processed_proposals = []
        
        try:
            while self.automation_proposal_queue:
                proposal = self.automation_proposal_queue.popleft()
                
                # Singleton evaluates the proposal using its LLM brain
                decision = await self._evaluate_automation_proposal(proposal)
                
                processed_proposals.append({
                    'proposal': proposal,
                    'decision': decision
                })
            
            return processed_proposals
            
        except Exception as e:
            logger.error(f"Error processing automation proposals: {e}")
            return processed_proposals
    
    async def _evaluate_automation_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Singleton evaluates an automation proposal and decides whether to approve.
        This is the KEY DECISION POINT - Singleton uses intelligence, not rules.
        """
        try:
            if not self.llm:
                logger.warning("No LLM brain available for proposal evaluation")
                return {'approved': False, 'reasoning': 'No LLM available'}
            
            # Build rich context for Singleton's decision
            prompt = f"""You are the Singleton evaluating an automation proposal.

PROPOSAL DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rule: {proposal.get('rule_name')}
Domain: {proposal.get('domain')}
Priority: {proposal.get('priority')}
Level: {proposal.get('level')}

TRIGGER REASON:
{proposal.get('trigger_reason')}

PROPOSED ACTIONS:
{chr(10).join(f"  - {action.get('type')}: {action.get('target')} via {action.get('method')}" for action in proposal.get('proposed_actions', []))}

SAFETY CONSTRAINTS:
{chr(10).join(f"  - {constraint}" for constraint in proposal.get('safety_constraints', []))}

SAFETY STATUS:
- Safety Framework Approved: {proposal.get('safety_approved')}
- Effectiveness Score: {proposal.get('effectiveness_score', 0.0):.1%}
- Max Executions/Hour: {proposal.get('max_executions_per_hour')}

CURRENT CONTEXT:
- Active Goals: {len(self.system_state.active_goals)}
- Active Tasks: {len(self.system_state.active_tasks)}
- System Mode: {self.system_state.mode.value}
- Recent Actions: {self.stats.get('tasks_completed', 0)} tasks completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECISION FRAMEWORK:
Consider:
1. Urgency: Is this CRITICAL priority? Does it need immediate action?
2. Safety: Are the safety constraints sufficient? Any risks?
3. Timing: Did you recently deploy changes? Is system stable?
4. Alignment: Does this align with your current goals and responsibilities?
5. Resource Impact: Will this interfere with active tasks?

Make your decision:
DECISION: APPROVE or REJECT
REASONING: [Your thought process in 2-3 sentences]
PRIORITY_ADJUSTMENT: [If approved, should priority be adjusted? CRITICAL/HIGH/NORMAL/LOW]
"""

            # Get Singleton's decision
            from core.services.unified_llm import LLMRequest
            llm_request = LLMRequest(
                prompt=prompt,
                system_prompt=self.llm.system_prompts.get("autonomous"),
                agent_type="autonomous",
                max_tokens=200,
                temperature=0.7,  # Allow some creativity in decision making
                enable_thinking=True  # Enable thinking mode for better reasoning
            )
            
            response = await self.llm.process_request(llm_request)
            decision_text = response.text

            # 🧠 CAPTURE COGNITIVE EXPERIENCE (Corrected Architecture) - Autonomous decision-making
            if hasattr(response, 'thinking_content') and response.thinking_content:
                try:
                    from core.agents.memory_agent import get_memory_agent
                    import time
                    import psutil

                    # Capture system metrics
                    process = psutil.Process()
                    active_task_count = len(self.active_tasks) if hasattr(self, 'active_tasks') else 0

                    system_metrics = {
                        "cpu_percent": psutil.cpu_percent(interval=0.1),
                        "memory_percent": process.memory_percent(),
                        "active_tasks": active_task_count,
                        "timestamp": time.time()
                    }

                    # Determine impact and irreversibility
                    decision_approved = "APPROVE" in decision_text.upper()
                    impact = "high"  # Automation decisions have high impact

                    # Automation approvals are often irreversible (creates new patterns)
                    irreversibility = "MOSTLY_IRREVERSIBLE" if decision_approved else "REVERSIBLE"

                    memory = get_memory_agent()

                    # Build task context for promotion evaluation
                    task_context = {
                        "task_id": f"automation_decision_{proposal.get('task_id', 'unknown')}_{int(time.time())}",
                        "domain": "autonomous_coordination",
                        "task_type": "automation_proposal_evaluation",
                        "impact": impact,  # High impact decisions
                        "irreversibility": irreversibility,
                        "novelty": 0.7,  # Automation proposals are novel
                        "memory_type": "episodic",
                        "proposal_name": proposal.get('rule_name'),
                        "proposal_type": proposal.get('type'),
                        "decision": "APPROVED" if decision_approved else "REJECTED",
                        "tags": ["autonomous", "decision_making", "automation", "coordination"],
                        "emotional_context": {
                            "task_load": "high" if active_task_count > 5 else "normal",
                            "confidence": "evaluating"
                        },
                        "decision_factors": {
                            "evaluation_criteria": ["impact", "risk", "timing", "alignment", "resources"],
                            "agent_type": "autonomous",
                            "active_task_count": active_task_count
                        }
                    }

                    # Process cognitive experience (will likely promote as PRECEDENT - high impact + novel)
                    memory_id = await memory.process_cognitive_experience(
                        raw_thinking=response.thinking_content,
                        task_context=task_context,
                        outcome=None,
                        system_metrics=system_metrics
                    )

                    if memory_id:
                        logger.info(f"✅ Promoted automation decision to memory: {memory_id} (scope=PRECEDENT)")
                    else:
                        logger.debug("⏭️ Automation decision discarded (unexpected)")

                except Exception as e:
                    logger.error(f"Failed to process cognitive experience: {e}")

            # Parse decision
            approved = "APPROVE" in decision_text.upper()
            
            # Extract reasoning
            reasoning_match = decision_text.split("REASONING:", 1)
            reasoning = reasoning_match[1].split("PRIORITY_ADJUSTMENT:")[0].strip() if len(reasoning_match) > 1 else "No reasoning provided"
            
            # Extract priority adjustment
            priority_match = decision_text.split("PRIORITY_ADJUSTMENT:", 1)
            priority_adjustment = priority_match[1].strip().split()[0] if len(priority_match) > 1 else None
            
            logger.info(f"🧠 Singleton decision on {proposal.get('rule_name')}: {'APPROVED' if approved else 'REJECTED'}")
            logger.info(f"   💭 Reasoning: {reasoning[:150]}...")
            
            # Store automation framework reference if we have it
            automation_framework = None
            if 'automation_framework' in self.config:
                automation_framework = self.config['automation_framework']
            
            # Communicate decision to automation framework
            if automation_framework and hasattr(automation_framework, 'approve_automation_task'):
                await automation_framework.approve_automation_task(
                    task_id=proposal.get('task_id'),
                    approved=approved,
                    reasoning=reasoning
                )
            
            return {
                'approved': approved,
                'reasoning': reasoning,
                'priority_adjustment': priority_adjustment,
                'thinking_content': response.thinking_content if hasattr(response, 'thinking_content') else None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating automation proposal: {e}")
            return {'approved': False, 'reasoning': f'Error during evaluation: {str(e)}'}
    
    async def _perception_phase(self):
        """Handle perception phase of coordination cycle"""
        try:
            async with self.watchdog.monitored_operation("perception_phase"):
                # Process health events from monitoring systems FIRST
                health_events = await self._process_health_events()
                if health_events:
                    logger.info(f"📊 Processed {len(health_events)} health events in perception")
                
                # Process automation proposals from ASI Automation Framework
                automation_proposals = await self._process_automation_proposals()
                if automation_proposals:
                    logger.info(f"🤖 Processed {len(automation_proposals)} automation proposals")
                
                # Get recent perceptions
                recent_perceptions = await self.perception.get_recent_perceptions(5)
                
                # Update system state with perception data
                if recent_perceptions:
                    self.system_state.performance_metrics["perception_active"] = 1.0
            
        except Exception as e:
            logger.error(f"Error in perception phase: {e}")

            import traceback
            # Log error with full details
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='_perception_phase',
                stack_trace=traceback.format_exc(),
                context={}
            )

    async def _planning_phase(self):
        """Handle planning phase of coordination cycle - Enhanced with intrinsic motivation"""
        try:
            async with self.watchdog.monitored_operation("planning_phase"):
                # Detect idle (no user interactions) using LLM metrics
                try:
                    llm_requests = 0
                    if self.llm and hasattr(self.llm, 'performance_metrics'):
                        llm_requests = int(self.llm.performance_metrics.get('requests_processed', 0))
                    if llm_requests == self._last_requests_processed:
                        self._idle_cycles += 1
                    else:
                        self._idle_cycles = 0
                    self._last_requests_processed = llm_requests
                except Exception:
                    pass
                # Check if we should generate curiosity-driven goals
                active_goal_count = len(self.system_state.active_goals)
                max_goals = self.config.get("max_concurrent_goals", 8)  # Increased from 5 to 8

                # Generate goals ONLY when idle for 2+ cycles (boredom detection)
                # Intrinsic motivation should only kick in when system detects boredom
                if active_goal_count < max_goals and self._idle_cycles >= 2:
                    skill_recommendations = await self.intrinsic_motivation.get_skill_recommendations()

                    # More aggressive goal generation - lower threshold from 2 to 3
                    if skill_recommendations or active_goal_count < 3:
                        new_goals = await self.generate_curiosity_driven_goals(
                            max_goals=max(2, max_goals - active_goal_count)  # Generate at least 2 goals
                        )

                        if new_goals:
                            logger.info(f"🎯 Generated {len(new_goals)} curiosity-driven goals")

                            # Log autonomous decision to generate goals
                            self.log_db.log_autonomous(
                                operation_type='goal_generation',
                                decision='generate_curiosity_driven_goals',
                                reasoning=f'Active goals ({active_goal_count}) below max ({max_goals}), idle cycles: {self._idle_cycles}',
                                confidence=0.8,
                                outcome=f'Generated {len(new_goals)} new goals',
                                metadata={
                                    'goals_generated': len(new_goals),
                                    'active_goal_count': active_goal_count,
                                    'idle_cycles': self._idle_cycles,
                                    'cycle_number': self.stats["cycles_completed"],
                                    'skill_recommendations': len(skill_recommendations) if skill_recommendations else 0
                                }
                            )
                        elif self._idle_cycles >= 2:
                            logger.info("😐 System idle detected (no user activity). Attempted curiosity-driven goals but none were created this cycle.")

                            # Log autonomous decision (no goals generated)
                            self.log_db.log_autonomous(
                                operation_type='goal_generation',
                                decision='attempted_goal_generation',
                                reasoning=f'System idle ({self._idle_cycles} cycles), attempted goal generation',
                                confidence=0.5,
                                outcome='No goals generated this cycle',
                                metadata={
                                    'goals_generated': 0,
                                    'active_goal_count': active_goal_count,
                                    'idle_cycles': self._idle_cycles,
                                    'cycle_number': self.stats["cycles_completed"]
                                }
                            )
                
                # Generate plans for goals without plans
                for goal_id in self.system_state.active_goals:
                    # Check if goal already has an active plan
                    planning_status = await self.planning.get_planning_status()
                    
                    # Generate plan if needed
                    plan = await self.planning.generate_plan(goal_id, {
                        "system_state": self.system_state,
                        "available_resources": self.system_state.resources
                    })
                    
                    if plan:
                        logger.debug(f"Generated plan for goal {goal_id}")
                
                # Calculate impact reward for planning activities
                if active_goal_count > 0:
                    impact_reward = await self.intrinsic_motivation.calculate_impact_reward({
                        "users_affected": 0,
                        "quality_improvement": 0.3,
                        "problem_solved": False,
                        "knowledge_shared": False
                    })
            
        except Exception as e:
            logger.error(f"Error in planning phase: {e}")

            import traceback
            # Log error with full details
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='_planning_phase',
                stack_trace=traceback.format_exc(),
                context={
                    'active_goals': len(self.system_state.active_goals),
                    'idle_cycles': self._idle_cycles,
                    'cycle_number': self.stats.get("cycles_completed", 0)
                }
            )
    
    async def _execution_phase(self):
        """
        Handle execution phase - Execute tasks using Singleton's capabilities.
        
        Tasks are mapped to actual Singleton methods:
        - RESEARCH tasks → _autonomous_research()
        - LEARNING tasks → _autonomous_learning()
        - SELF_IMPROVEMENT tasks → _autonomous_self_improvement()
        - MEMORY tasks → _autonomous_memory_consolidation()
        """
        try:
            # Get next tasks to execute
            next_tasks = await self.planning.get_next_tasks(self.system_state)
            
            if not next_tasks:
                logger.debug("No tasks ready for execution this cycle")
                return
            
            # Execute tasks that fit current capacity
            execution_status = await self.execution.get_execution_status()
            available_capacity = execution_status.get("available_capacity", 3)
            
            tasks_to_execute = next_tasks[:available_capacity]
            
            for task in tasks_to_execute:
                try:
                    logger.info(f"▶️  Executing task: {task.description} (type: {task.type.value})")
                    
                    # Map task type to Singleton capability
                    success = await self._execute_task_with_singleton(task)
                    
                    if success:
                        # Update task status in planning
                        await self.planning.update_task_status(
                            task.id, 
                            TaskStatus.COMPLETED,
                            result={'success': True, 'executed_by': 'singleton'}
                        )
                        self.stats["tasks_completed"] += 1
                        logger.info(f"✅ Task completed: {task.description[:60]}...")

                        # Send Slack notification for task milestones (every 100 tasks)
                        if self.stats["tasks_completed"] % 100 == 0:
                            try:
                                await self.slack_notifier.send_learning_milestone(
                                    milestone_title=f"{self.stats['tasks_completed']} Tasks Completed",
                                    milestone_description=f"Autonomous coordinator reached {self.stats['tasks_completed']} completed tasks milestone",
                                    metrics={
                                        'tasks_completed': self.stats['tasks_completed'],
                                        'goals_achieved': self.stats.get('goals_achieved', 0),
                                        'cycles_completed': self.stats.get('cycles_completed', 0),
                                        'system_efficiency': self.stats.get('system_efficiency', 0.0),
                                        'cross_domain_operations': self.stats.get('cross_domain_operations', 0),
                                        'uptime_seconds': self.stats.get('uptime_seconds', 0.0)
                                    }
                                )
                            except Exception as e:
                                logger.warning(f"Failed to send task milestone notification: {e}")
                    else:
                        await self.planning.update_task_status(
                            task.id,
                            TaskStatus.FAILED,
                            result={'success': False, 'reason': 'execution_failed'}
                        )
                        logger.warning(f"❌ Task failed: {task.description[:60]}...")
                        
                except Exception as task_error:
                    logger.error(f"Error executing task {task.id}: {task_error}")

                    import traceback
                    # Log error with full details
                    self.log_db.log_error(
                        error_type=type(task_error).__name__,
                        error_message=str(task_error),
                        module='autonomous_coordinator',
                        function='_execution_phase',
                        stack_trace=traceback.format_exc(),
                        context={'task_id': task.id, 'task_description': task.description[:200]}
                    )

                    # Send Slack alert for task execution failure
                    try:
                        # Build detailed failure message
                        task_type_str = task.type.value if hasattr(task.type, 'value') else str(task.type)
                        error_details = (
                            f"**Task execution failed**\n\n"
                            f"**Task ID**: {task.id}\n"
                            f"**Task Type**: {task_type_str}\n"
                            f"**Description**: {task.description}\n\n"
                            f"**Error Type**: {type(task_error).__name__}\n"
                            f"**Error Message**: {str(task_error)}\n\n"
                            f"**System Stats**:\n"
                            f"   - Tasks Completed: {self.stats['tasks_completed']}\n"
                            f"   - Cycles Completed: {self.stats['cycles_completed']}"
                        )

                        await self.slack_notifier.send_security_alert(
                            alert_title="Task Execution Failure",
                            alert_message=error_details,
                            severity="MODERATE",
                            metadata={
                                'error': str(task_error),
                                'error_type': type(task_error).__name__,
                                'task_id': task.id,
                                'task_type': task_type_str,
                                'task_description': task.description,
                                'tasks_completed': self.stats['tasks_completed'],
                                'cycles_completed': self.stats['cycles_completed']
                            }
                        )
                    except Exception as notify_error:
                        logger.warning(f"Failed to send task failure notification: {notify_error}")

            # Check for completed tasks
            await self._check_task_completions()

        except Exception as e:
            logger.error(f"Error in execution phase: {e}")

            import traceback
            # Log error with full details
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='_execution_phase',
                stack_trace=traceback.format_exc(),
                context={'active_tasks': len(self.system_state.active_tasks)}
            )
    
    async def _execute_task_with_singleton(self, task: Task) -> bool:
        """
        Execute a task using the appropriate Singleton capability.

        This is the KEY CONNECTION: Tasks become actual function calls.
        """
        if not self.llm:
            logger.warning("No LLM brain available for task execution")
            return False

        import time
        start_time = time.time()

        # Log task execution start with task_id
        self.log_db.log_coordination(
            coordinator_type='autonomous',
            action='task_execution_start',
            task_id=task.id,
            status='executing',
            metadata={
                'task_type': task.type.value,
                'task_description': task.description[:200],
                'priority': task.priority
            }
        )

        try:
            # Map task type to Singleton method
            task_type = task.type
            
            if task_type == TaskType.RESEARCH:
                if hasattr(self.llm, '_autonomous_research'):
                    await self.llm._autonomous_research()
                    execution_time = time.time() - start_time

                    # Log successful execution
                    self.log_db.log_coordination(
                        coordinator_type='autonomous',
                        action='task_execution_complete',
                        task_id=task.id,
                        status='completed',
                        result='Research task executed successfully',
                        metadata={'task_type': task_type.value, 'execution_time': execution_time}
                    )

                    # Log performance metrics
                    self.log_db.log_performance(
                        operation='task_execution',
                        duration=execution_time,
                        success=True,
                        details={'task_id': task.id, 'task_type': task_type.value}
                    )
                    return True

            elif task_type == TaskType.ANALYSIS or task_type == TaskType.LEARNING:
                # Alternate between learning and reasoning experiments
                if hasattr(self.llm, '_autonomous_learning') and hasattr(self.llm, '_experiment_with_reasoning_methods'):
                    # 50% chance: traditional learning, 50% chance: reasoning experiments
                    import random
                    method = 'learning' if random.random() < 0.5 else 'reasoning_experiments'
                    if method == 'learning':
                        await self.llm._autonomous_learning()
                    else:
                        await self.llm._experiment_with_reasoning_methods()

                    execution_time = time.time() - start_time

                    # Log successful execution
                    self.log_db.log_coordination(
                        coordinator_type='autonomous',
                        action='task_execution_complete',
                        task_id=task.id,
                        status='completed',
                        result=f'Analysis task executed successfully ({method})',
                        metadata={'task_type': task_type.value, 'method': method, 'execution_time': execution_time}
                    )

                    # Log performance metrics
                    self.log_db.log_performance(
                        operation='task_execution',
                        duration=execution_time,
                        success=True,
                        details={'task_id': task.id, 'task_type': task_type.value, 'method': method}
                    )
                    return True
                elif hasattr(self.llm, '_autonomous_learning'):
                    await self.llm._autonomous_learning()
                    execution_time = time.time() - start_time

                    # Log successful execution
                    self.log_db.log_coordination(
                        coordinator_type='autonomous',
                        action='task_execution_complete',
                        task_id=task.id,
                        status='completed',
                        result='Learning task executed successfully',
                        metadata={'task_type': task_type.value, 'execution_time': execution_time}
                    )

                    # Log performance metrics
                    self.log_db.log_performance(
                        operation='task_execution',
                        duration=execution_time,
                        success=True,
                        details={'task_id': task.id, 'task_type': task_type.value}
                    )
                    return True

            elif task_type == TaskType.SYNTHESIS or task_type == TaskType.PLANNING:
                # Self-improvement for synthesis/planning tasks
                if hasattr(self.llm, '_autonomous_self_improvement'):
                    await self.llm._autonomous_self_improvement()
                    execution_time = time.time() - start_time

                    # Log successful execution
                    self.log_db.log_coordination(
                        coordinator_type='autonomous',
                        action='task_execution_complete',
                        task_id=task.id,
                        status='completed',
                        result='Synthesis/planning task executed successfully',
                        metadata={'task_type': task_type.value, 'execution_time': execution_time}
                    )

                    # Log performance metrics
                    self.log_db.log_performance(
                        operation='task_execution',
                        duration=execution_time,
                        success=True,
                        details={'task_id': task.id, 'task_type': task_type.value}
                    )
                    return True

            elif task_type == TaskType.VALIDATION:
                # Memory consolidation for validation tasks
                if hasattr(self.llm, '_autonomous_memory_consolidation'):
                    await self.llm._autonomous_memory_consolidation()
                    execution_time = time.time() - start_time

                    # Log successful execution
                    self.log_db.log_coordination(
                        coordinator_type='autonomous',
                        action='task_execution_complete',
                        task_id=task.id,
                        status='completed',
                        result='Validation task executed successfully',
                        metadata={'task_type': task_type.value, 'execution_time': execution_time}
                    )

                    # Log performance metrics
                    self.log_db.log_performance(
                        operation='task_execution',
                        duration=execution_time,
                        success=True,
                        details={'task_id': task.id, 'task_type': task_type.value}
                    )
                    return True

            else:
                # Default: Use execution controller's task handler
                result = await self.execution.execute_task(task)
                execution_time = time.time() - start_time

                # Log execution result
                self.log_db.log_coordination(
                    coordinator_type='autonomous',
                    action='task_execution_complete',
                    task_id=task.id,
                    status='completed' if result else 'failed',
                    result=f'Task handled by execution controller: {result}',
                    metadata={'task_type': task_type.value, 'execution_time': execution_time}
                )

                # Log performance metrics
                self.log_db.log_performance(
                    operation='task_execution',
                    duration=execution_time,
                    success=result,
                    details={'task_id': task.id, 'task_type': task_type.value}
                )
                return result

            # If we get here, task type not mapped
            execution_time = time.time() - start_time
            logger.warning(f"Task type {task_type} not mapped to Singleton capability")

            # Log unmapped task type
            self.log_db.log_coordination(
                coordinator_type='autonomous',
                action='task_execution_failed',
                task_id=task.id,
                status='failed',
                result=f'Task type {task_type} not mapped to capability',
                metadata={'task_type': task_type.value, 'execution_time': execution_time}
            )

            return False

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing task with singleton: {e}")

            import traceback
            # Log error with full details
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='_execute_task_with_singleton',
                stack_trace=traceback.format_exc(),
                context={'task_id': task.id, 'task_type': task.type.value, 'task_description': task.description}
            )

            # Log failed coordination
            self.log_db.log_coordination(
                coordinator_type='autonomous',
                action='task_execution_failed',
                task_id=task.id,
                status='error',
                result=f'Exception during execution: {str(e)}',
                metadata={'task_type': task.type.value, 'execution_time': execution_time, 'error': type(e).__name__}
            )

            # Send Slack alert for singleton task execution error
            try:
                await self.slack_notifier.send_security_alert(
                    alert_title="Singleton Task Execution Error",
                    alert_message=f"Critical error in task execution: {task.description[:150]}\nError: {str(e)[:200]}",
                    severity="HIGH",
                    metadata={
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'task_id': task.id,
                        'task_type': task.type.value,
                        'execution_time': execution_time,
                        'tasks_completed': self.stats['tasks_completed']
                    }
                )
            except Exception as notify_error:
                logger.warning(f"Failed to send singleton task error notification: {notify_error}")

            return False
    
    async def _learning_phase(self):
        """Handle learning phase of coordination cycle - Enhanced with intrinsic motivation"""
        try:
            # Get learning recommendations
            context = {
                "system_mode": self.system_state.mode.value,
                "active_goals": len(self.system_state.active_goals),
                "resource_usage": self.system_state.resource_usage
            }
            
            recommendations = await self.learning.get_recommendations(context)
            
            # Apply high-confidence recommendations and calculate intrinsic rewards
            applied_recommendations = []
            total_intrinsic_reward = 0.0
            
            for rec in recommendations:
                if rec.get("confidence", 0) > 0.8:
                    success = await self._apply_learning_recommendation(rec)
                    if success:
                        applied_recommendations.append(rec)
                        
                        # Calculate competence reward for successful learning application
                        competence_reward = await self.intrinsic_motivation.calculate_competence_reward(
                            skill_name=rec.get("action", {}).get("type", "general_learning"),
                            performance=rec.get("confidence", 0.8),
                            success=True
                        )
                        total_intrinsic_reward += competence_reward.reward_value
            
            # Identify exploration targets from perception data
            perception_stats = await self.perception.get_statistics()
            if perception_stats.get("novel_patterns", 0) > 0:
                # Calculate curiosity reward for discovering novel patterns
                curiosity_reward = await self.intrinsic_motivation.calculate_curiosity_reward({
                    "information_gain": min(1.0, perception_stats.get("novel_patterns", 0) / 10.0),
                    "uncertainty_reduction": 0.5,
                    "question_complexity": 0.6,
                    "answer_depth": 0.5
                })
                total_intrinsic_reward += curiosity_reward.reward_value
            
            # Calculate novelty reward for current cycle
            cycle_experience = {
                "active_goals": len(self.system_state.active_goals),
                "active_tasks": len(self.system_state.active_tasks),
                "cycle_count": self.stats["cycles_completed"],
                "resource_usage": self.system_state.resource_usage
            }
            novelty_reward = await self.intrinsic_motivation.calculate_novelty_reward(cycle_experience)
            total_intrinsic_reward += novelty_reward.reward_value
            
            # Calculate autonomy reward (coordination is self-directed)
            autonomy_reward = await self.intrinsic_motivation.calculate_autonomy_reward({
                "self_initiated": True,
                "choice_made": len(recommendations) > 0,
                "exploration_ratio": 0.5  # Balanced exploration/exploitation
            })
            total_intrinsic_reward += autonomy_reward.reward_value
            
            # Get top exploration targets for next cycle
            exploration_targets = await self.intrinsic_motivation.get_top_exploration_targets(limit=5)
            
            # Store learning insights in memory with intrinsic reward information
            if applied_recommendations or total_intrinsic_reward > 0.1:
                await self.store_memory(
                    MemoryType.PROCEDURAL,
                    {
                        "event": "learning_with_intrinsic_rewards",
                        "recommendations": applied_recommendations,
                        "total_intrinsic_reward": total_intrinsic_reward,
                        "exploration_targets": [t.description for t in exploration_targets],
                        "context": context,
                        "timestamp": datetime.now().isoformat()
                    },
                    importance=0.8 + (total_intrinsic_reward * 0.2),  # Boost importance with intrinsic reward
                    tags=["learning", "intrinsic_motivation", "autonomous_cycle"]
                )
            
            # Log intrinsic motivation insights
            if total_intrinsic_reward > 0.5:
                logger.info(f"🌟 High intrinsic motivation cycle! Total reward: {total_intrinsic_reward:.2f}")
            
        except Exception as e:
            logger.error(f"Error in learning phase: {e}")

            import traceback
            # Log error with full details
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='_learning_phase',
                stack_trace=traceback.format_exc(),
                context={}
            )

    async def _execute_registered_capabilities(self):
        """
        Execute registered system capabilities based on conditions and intervals.
        
        This method enables TRUE ADAPTIVE INTELLIGENCE by allowing the coordinator to decide
        when capabilities should run based on system state, not hardcoded timers.
        
        Capabilities are checked in priority order (critical > high > medium > low) and
        executed only if:
        1. Minimum interval has elapsed since last run
        2. All configured conditions are met (feedback samples, performance, etc.)
        
        This transforms rigid "run every N seconds" into intelligent "run when needed"
        based on system feedback, performance metrics, and resource availability.
        """
        if not hasattr(self, 'registered_capabilities') or not self.registered_capabilities:
            return  # No capabilities registered yet
        
        try:
            now = datetime.now()
            
            # Sort capabilities by priority
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            sorted_capabilities = sorted(
                self.registered_capabilities.items(),
                key=lambda x: priority_order.get(x[1]['priority'], 4)
            )
            
            for cap_name, cap_config in sorted_capabilities:
                try:
                    # Skip if not active
                    if cap_config['status'] != 'active':
                        continue
                    
                    # Check interval - has enough time passed?
                    last_run = self.capability_last_run.get(cap_name, datetime.min)
                    elapsed = (now - last_run).total_seconds()
                    interval = cap_config['interval']
                    
                    if elapsed < interval:
                        continue  # Not time yet
                    
                    # Check all conditions
                    conditions = cap_config['conditions']
                    if not await self._check_capability_conditions(cap_name, conditions):
                        logger.debug(f"Capability '{cap_name}' conditions not met, skipping")
                        continue
                    
                    # Execute the capability
                    instance = cap_config['instance']
                    method_name = cap_config['method']
                    method = getattr(instance, method_name)

                    logger.info(f"🔧 Executing capability: {cap_name} (priority: {cap_config['priority']})")

                    # Log capability execution start
                    self.log_db.log_coordination(
                        coordinator_type='autonomous',
                        action='capability_execution_start',
                        status='executing',
                        metadata={
                            'capability': cap_name,
                            'priority': cap_config['priority'],
                            'elapsed_since_last': elapsed,
                            'execution_count': cap_config['execution_count']
                        }
                    )

                    import time
                    start_time = time.time()

                    # Call the method (handle both async and sync)
                    if asyncio.iscoroutinefunction(method):
                        result = await method()
                    else:
                        result = method()

                    execution_time = time.time() - start_time

                    # Update tracking
                    self.capability_last_run[cap_name] = now
                    cap_config['execution_count'] += 1
                    cap_config['last_result'] = result
                    cap_config['last_error'] = None

                    logger.info(f"✅ Capability '{cap_name}' executed successfully (run #{cap_config['execution_count']})")

                    # Log successful capability execution
                    self.log_db.log_coordination(
                        coordinator_type='autonomous',
                        action='capability_execution_complete',
                        status='completed',
                        result=f"Capability '{cap_name}' executed successfully",
                        metadata={
                            'capability': cap_name,
                            'priority': cap_config['priority'],
                            'execution_count': cap_config['execution_count'],
                            'execution_time': execution_time
                        }
                    )

                    # Log performance metrics
                    self.log_db.log_performance(
                        operation='capability_execution',
                        duration=execution_time,
                        success=True,
                        details={
                            'capability': cap_name,
                            'priority': cap_config['priority'],
                            'execution_count': cap_config['execution_count']
                        }
                    )

                    # Store execution in memory for learning
                    await self.store_memory(
                        MemoryType.PROCEDURAL,
                        {
                            'event': 'capability_execution',
                            'capability': cap_name,
                            'priority': cap_config['priority'],
                            'result': str(result)[:500] if result else None,  # Truncate large results
                            'execution_count': cap_config['execution_count'],
                            'execution_time': execution_time,
                            'timestamp': now.isoformat()
                        },
                        importance=0.7 if cap_config['priority'] in ['critical', 'high'] else 0.5,
                        tags=['capability', 'autonomous', cap_name]
                    )

                except Exception as e:
                    logger.error(f"Error executing capability '{cap_name}': {e}")
                    cap_config['last_error'] = str(e)
                    cap_config['status'] = 'error'

                    import traceback
                    # Log error with full details
                    self.log_db.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        module='autonomous_coordinator',
                        function='_execute_registered_capabilities',
                        stack_trace=traceback.format_exc(),
                        context={
                            'capability': cap_name,
                            'priority': cap_config['priority'],
                            'method': method_name
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error in capability execution phase: {e}")

            import traceback
            # Log error with full details
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='_execute_registered_capabilities',
                stack_trace=traceback.format_exc(),
                context={'registered_capabilities_count': len(self.registered_capabilities) if hasattr(self, 'registered_capabilities') else 0}
            )

    async def _check_capability_conditions(self, cap_name: str, conditions: Dict[str, Any]) -> bool:
        """
        Check if all conditions for a capability are met.
        
        Args:
            cap_name: Capability name (for logging)
            conditions: Dict of condition checks
        
        Returns:
            True if all conditions met, False otherwise
        """
        try:
            # Check feedback sample minimum
            if 'min_feedback_samples' in conditions:
                min_samples = conditions['min_feedback_samples']
                # Query feedback count from memory using MemoryQuery
                try:
                    from core.memory import MemoryQuery, MemoryType
                    
                    feedback_query = MemoryQuery(
                        query_id=f"capability_check_{cap_name}_{datetime.now().timestamp()}",
                        content="user feedback and ratings",
                        memory_types=[MemoryType.EPISODIC],
                        max_results=min_samples + 10,  # Fetch a bit more to ensure we get enough
                        min_confidence=0.0
                    )
                    result = await self.memory.search_memories(feedback_query)
                    
                    # Filter for feedback-related memories
                    feedback_count = sum(1 for m in result.memories if 'feedback' in str(m.content).lower() or 'rating' in str(m.content).lower())
                    
                    if feedback_count < min_samples:
                        logger.debug(f"Capability '{cap_name}': Insufficient feedback samples ({feedback_count}/{min_samples})")
                        return False
                except Exception as e:
                    logger.debug(f"Could not query feedback memories: {e}")
                    # If we can't check, allow execution (fail open for this condition)
            
            # Check performance threshold
            if 'performance_threshold' in conditions:
                threshold = conditions['performance_threshold']
                # Resource usage might be a dict or float
                resource_usage = self.system_state.resource_usage
                if isinstance(resource_usage, dict):
                    current_performance = resource_usage.get('system_health', 1.0)
                else:
                    current_performance = 1.0  # Assume healthy if no data
                
                if current_performance < threshold:
                    logger.debug(f"Capability '{cap_name}': Performance below threshold ({current_performance:.2f}/{threshold})")
                    return False
            
            # Check error rate
            if 'error_rate_max' in conditions:
                max_error_rate = conditions['error_rate_max']
                current_error_rate = self.stats.get('error_rate', 0.0)
                
                if current_error_rate > max_error_rate:
                    logger.debug(f"Capability '{cap_name}': Error rate too high ({current_error_rate:.2f}/{max_error_rate})")
                    return False
            
            # Check memory usage
            if 'memory_usage_max' in conditions:
                max_memory = conditions['memory_usage_max']
                resource_usage = self.system_state.resource_usage
                if isinstance(resource_usage, dict):
                    current_memory = resource_usage.get('memory_percent', 0.0)
                else:
                    current_memory = 0.0  # Assume OK if no data
                
                if current_memory > max_memory:
                    logger.debug(f"Capability '{cap_name}': Memory usage too high ({current_memory:.2f}/{max_memory})")
                    return False
            
            # Custom check function
            if 'custom_check' in conditions:
                check_func = conditions['custom_check']
                if callable(check_func):
                    if asyncio.iscoroutinefunction(check_func):
                        result = await check_func(self)
                    else:
                        result = check_func(self)
                    
                    if not result:
                        logger.debug(f"Capability '{cap_name}': Custom check failed")
                        return False
            
            return True  # All conditions met
            
        except Exception as e:
            logger.error(f"Error checking conditions for '{cap_name}': {e}")
            return False  # Fail safe - don't execute if condition check fails
    
    async def _check_task_completions(self):
        """Check for task completions and update system state"""
        try:
            execution_status = await self.execution.get_execution_status()

            # Get completed tasks from execution controller
            completed_count = execution_status.get("completed_tasks", 0)

            # Check which of our active tasks have completed
            completed_task_ids = []

            # The execution controller maintains completed_tasks dict
            # We need to check if our active tasks are in there
            for task_id in list(self.system_state.active_tasks):
                # Check with execution controller if task completed
                # Completed tasks are moved from running_tasks to completed_tasks
                if task_id in self.execution.completed_tasks:
                    completed_task = self.execution.completed_tasks[task_id]

                    # Verify it actually completed successfully
                    if completed_task.status == TaskStatus.COMPLETED:
                        completed_task_ids.append(task_id)

                        # Log completion details
                        execution_time = "N/A"
                        execution_time_seconds = 0.0
                        if completed_task.completed_at and completed_task.created_at:
                            execution_time_seconds = (completed_task.completed_at - completed_task.created_at).total_seconds()
                            execution_time = f"{execution_time_seconds:.2f}s"

                        logger.info(
                            f"✅ Task completed: {completed_task.description} "
                            f"(execution time: {execution_time})"
                        )

                        # Record to constitutional framework if relevant
                        if hasattr(completed_task, 'result') and completed_task.result:
                            quality_score = completed_task.result.get('quality_score', 0.8)
                        else:
                            quality_score = 0.8  # Default for successful tasks

                        # Log task completion coordination
                        self.log_db.log_coordination(
                            coordinator_type='autonomous',
                            action='task_completion_verified',
                            task_id=task_id,
                            status='completed',
                            result=f'Task verified as completed: {completed_task.description[:100]}',
                            metadata={
                                'execution_time': execution_time_seconds,
                                'quality_score': quality_score,
                                'task_type': completed_task.type.value if hasattr(completed_task, 'type') else 'unknown'
                            }
                        )

                    elif completed_task.status == TaskStatus.FAILED:
                        # Remove from active but don't count as completion
                        completed_task_ids.append(task_id)
                        logger.warning(f"❌ Task failed: {completed_task.description}")

                        # Log task failure coordination
                        self.log_db.log_coordination(
                            coordinator_type='autonomous',
                            action='task_completion_verified',
                            task_id=task_id,
                            status='failed',
                            result=f'Task verified as failed: {completed_task.description[:100]}',
                            metadata={
                                'task_type': completed_task.type.value if hasattr(completed_task, 'type') else 'unknown',
                                'failure_reason': completed_task.result.get('reason') if hasattr(completed_task, 'result') and completed_task.result else 'unknown'
                            }
                        )
                        self.stats["tasks_failed"] += 1
            
            # Update statistics and remove completed tasks
            for task_id in completed_task_ids:
                if task_id in self.system_state.active_tasks:
                    self.system_state.active_tasks.remove(task_id)
                    
                    # Only count successful completions
                    task = self.execution.completed_tasks[task_id]
                    if task.status == TaskStatus.COMPLETED:
                        self.stats["tasks_completed"] += 1
                        
                        # Update planning engine
                        await self.planning.update_task_status(
                            task_id, 
                            TaskStatus.COMPLETED,
                            task.result
                        )
            
            # Log summary if any tasks completed
            if completed_task_ids:
                logger.info(
                    f"Task completion cycle: {len(completed_task_ids)} tasks finished "
                    f"(successful: {self.stats['tasks_completed']}, "
                    f"failed: {self.stats['tasks_failed']})"
                )
            
        except Exception as e:
            logger.error(f"Error checking task completions: {e}")

            import traceback
            # Log error with full details
            self.log_db.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                module='autonomous_coordinator',
                function='_check_task_completions',
                stack_trace=traceback.format_exc(),
                context={'active_tasks_count': len(self.system_state.active_tasks)}
            )

    async def _analyze_for_goal_creation(self, perception_data: PerceptionData) -> Optional[str]:
        """Analyze perception data to determine if a new goal should be created"""
        try:
            # Simple heuristics for goal creation
            content = perception_data.content
            
            # Check for explicit requests
            if "request" in content or "goal" in content:
                description = content.get("text", f"Handle {perception_data.data_type} input")
                goal = await self.planning.create_goal(description, Priority.MEDIUM)
                if goal:
                    self.system_state.active_goals.append(goal.id)
                    return goal.id
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing for goal creation: {e}")
            return None
    
    async def _apply_learning_recommendation(self, recommendation: Dict[str, Any]):
        """Apply a learning recommendation to improve system performance"""
        try:
            action = recommendation.get("action", {})
            action_type = action.get("type", "unknown")
            
            logger.info(f"Applying learning recommendation: {action_type}")
            
            # Apply different types of recommendations
            if action_type == "adjust_cycle_interval":
                new_interval = action.get("value", self.coordination_cycle_interval)
                # Allow longer intervals (up to 1 hour) to support deep thinking cycles
                self.coordination_cycle_interval = max(1.0, min(3600.0, new_interval))
            
            elif action_type == "prioritize_task_type":
                task_type = action.get("task_type")
                priority_boost = action.get("priority_boost", 0.2)
                
                # Adjust task prioritization in planning engine
                # This increases priority for tasks of a specific type
                logger.info(f"Boosting priority for task type: {task_type} by {priority_boost}")
                
                # Get all active plans from planning engine
                for plan_id, plan in self.planning.active_plans.items():
                    for task in plan.tasks:
                        # Check if task matches the type to prioritize
                        if hasattr(task, 'type') and str(task.type).lower() == str(task_type).lower():
                            # Boost the task priority
                            current_priority = task.priority
                            
                            # Map priority to numeric, boost, then map back
                            priority_map = {
                                Priority.LOW: 1,
                                Priority.MEDIUM: 2,
                                Priority.HIGH: 3,
                                Priority.CRITICAL: 4
                            }
                            
                            priority_value = priority_map.get(current_priority, 2)
                            new_priority_value = min(4, priority_value + 1)  # Boost by one level
                            
                            # Reverse map back to Priority enum
                            reverse_map = {1: Priority.LOW, 2: Priority.MEDIUM, 3: Priority.HIGH, 4: Priority.CRITICAL}
                            task.priority = reverse_map.get(new_priority_value, Priority.HIGH)
                            
                            logger.info(
                                f"   Boosted task '{task.description[:50]}...' "
                                f"from {current_priority.name} to {task.priority.name}"
                            )
                    
                    # Update plan in database
                    await self.planning._store_plan(plan)
                
                # Also boost future goals related to this task type
                for goal_id, goal in self.planning.current_goals.items():
                    # Check if goal description relates to this task type
                    if task_type.lower() in goal.description.lower():
                        current_priority = goal.priority
                        
                        priority_map = {
                            Priority.LOW: 1,
                            Priority.MEDIUM: 2,
                            Priority.HIGH: 3,
                            Priority.CRITICAL: 4
                        }
                        
                        priority_value = priority_map.get(current_priority, 2)
                        new_priority_value = min(4, priority_value + 1)
                        
                        reverse_map = {1: Priority.LOW, 2: Priority.MEDIUM, 3: Priority.HIGH, 4: Priority.CRITICAL}
                        goal.priority = reverse_map.get(new_priority_value, Priority.HIGH)
                        
                        logger.info(
                            f"   Boosted goal '{goal.description[:50]}...' "
                            f"from {current_priority.name} to {goal.priority.name}"
                        )
                        
                        # Update goal in database
                        await self.planning._store_goal(goal)
                
                logger.info(f"✅ Task type '{task_type}' prioritization adjustment complete")
            
            elif action_type == "allocate_resources":
                resource_type = action.get("resource_type")
                allocation = action.get("allocation", 1.0)
                self.system_state.resources[resource_type] = allocation
            
        except Exception as e:
            logger.error(f"Error applying learning recommendation: {e}")
    
    async def _update_system_state(self):
        """Update current system state"""
        try:
            self.system_state.timestamp = datetime.now().timestamp()
            
            # Update resource usage (simplified calculation)
            execution_status = await self.execution.get_execution_status()
            self.system_state.resource_usage = execution_status.get("resource_usage", 0.0)
            
            # Update performance metrics
            perception_stats = await self.perception.get_statistics()
            planning_status = await self.planning.get_planning_status()
            
            self.system_state.performance_metrics.update({
                "perception_queue_length": perception_stats.get("queue_length", 0),
                "active_plans": planning_status.get("active_plans", 0),
                "pending_tasks": planning_status.get("pending_tasks", 0)
            })
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
    
    async def shutdown(self):
        """Shutdown the autonomous system gracefully"""
        logger.info("Shutting down autonomous system...")
        
        self.active = False
        
        # Cancel coordination cycle
        if self.coordination_task:
            self.coordination_task.cancel()
            try:
                await self.coordination_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown modules
        modules = [
            ("Learning Adapter", self.learning),
            ("Intrinsic Motivation System", self.intrinsic_motivation),
            ("Execution Controller", self.execution),
            ("Planning Engine", self.planning),
            ("Perception Manager", self.perception)
        ]
        
        for name, module in modules:
            try:
                await module.shutdown()
                logger.info(f"{name} shutdown completed")
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
        
        logger.info("Autonomous system shutdown completed")
    
    # =========================================================================
    # ENHANCED REASONING & LEARNING - Singleton's Unified Intelligence
    # =========================================================================
    
    def get_intelligence_capabilities(self) -> Dict[str, Any]:
        """
        Get all intelligence capabilities available to the Singleton
        
        Enhanced reasoning and learning systems accessible across the entire system
        """
        return {
            "abstract_reasoning": self.abstract_reasoning,
            "quantum_reasoning": self.quantum_reasoning,
            "proof_engine": self.proof_engine,
            "neural_bridge": self.neural_bridge,
            "unified_learning": self.unified_learning,
            "meta_learning": self.meta_learning,
            "causal_analyzer": self.causal_analyzer,
            "frontier_research": self.frontier_research,
            "status": {
                "abstract_reasoning_ready": self.abstract_reasoning is not None,
                "quantum_reasoning_ready": self.quantum_reasoning is not None,
                "proof_engine_ready": self.proof_engine is not None,
                "neural_bridge_ready": self.neural_bridge is not None,
                "unified_learning_ready": self.unified_learning is not None,
                "meta_learning_ready": self.meta_learning is not None,
                "causal_analysis_ready": self.causal_analyzer is not None,
                "frontier_research_ready": self.frontier_research is not None
            }
        }

    async def _receive_health_event(self, health_event: Dict[str, Any]):
        """
        Process health event with AI-powered analysis using Obsidian3.

        This method enables the Singleton to use its intelligence (Obsidian3-14B)
        to analyze system health issues, determine root causes, and execute
        autonomous recovery actions.

        Args:
            health_event: Health event from MonitoringCoordinator
        """
        try:
            logger.info(f"🧠 AI analyzing health event: {health_event.get('event_type', 'unknown')}")

            # Extract event details
            event_type = health_event.get('event_type', 'unknown')
            severity = health_event.get('severity', 'unknown')
            component = health_event.get('component', 'unknown')
            proposed_actions = health_event.get('proposed_actions', [])

            # Use Obsidian3 to analyze the health event
            analysis = await self._analyze_health_with_ai(health_event)

            if not analysis:
                logger.warning("AI health analysis returned no results")
                return

            # Determine if immediate action is required
            if analysis.get('immediate_action_required'):
                logger.warning(f"⚠️ IMMEDIATE ACTION REQUIRED for {component}")

                # Execute recovery if risk is acceptable
                best_action = analysis.get('recommended_action')
                if best_action and best_action.get('risk_level', 10) <= 5:
                    logger.info(f"🔧 Executing autonomous recovery: {best_action.get('name')}")
                    await self._execute_recovery(best_action, health_event)
                else:
                    logger.warning(f"⚠️ Recovery action too risky ({best_action.get('risk_level')}/10) - alerting administrator")

            else:
                logger.info(f"ℹ️ Health event logged, no immediate action required")

        except Exception as e:
            logger.error(f"Error processing health event: {e}")
            import traceback
            traceback.print_exc()

    async def _analyze_health_with_ai(self, health_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use Obsidian3 (Torin's brain) to analyze health events and recommend actions.

        This transforms passive monitoring into intelligent self-diagnosis by using
        the LLM to understand complex failure patterns and recommend recovery strategies.

        Args:
            health_event: Health event details

        Returns:
            AI analysis with recommended actions
        """
        try:
            from core.services.unified_llm import get_llm_service
            import json

            llm = await get_llm_service()

            # Create analysis prompt
            analysis_prompt = f"""You are analyzing your own system health. A health event has been detected.

HEALTH EVENT:
{json.dumps(health_event, indent=2)}

Analyze this event and provide:
1. Severity assessment (1-10 scale, where 10 is critical)
2. Root cause hypothesis (what likely caused this)
3. Recommended recovery action (choose the safest option)
4. Risk level of recommended action (1-10, where 10 is very risky)
5. Whether immediate action is required (true/false)
6. Predicted outcome if no action is taken

IMPORTANT: Be conservative. Only recommend immediate action if:
- Severity >= 7
- Risk level <= 5
- You are confident the action will help

Respond in JSON format with these exact keys:
{{
    "severity": <number 1-10>,
    "root_cause": "<brief explanation>",
    "recommended_action": {{
        "name": "<action name>",
        "description": "<what it does>",
        "parameters": {{}}
    }},
    "risk_level": <number 1-10>,
    "immediate_action_required": <true/false>,
    "predicted_outcome_no_action": "<what happens if we do nothing>"
}}"""

            response = await llm.generate(
                prompt=analysis_prompt,
                system_prompt=llm.system_prompts.get("health_analyst"),
                agent_type="health_analyst",
                max_tokens=1024,
                temperature=0.3  # Low temperature for consistent, reliable analysis
            )

            if not response or not response.text:
                logger.error("LLM returned empty response for health analysis")
                return None

            # Parse JSON response
            try:
                analysis = json.loads(response.text)
                logger.info(f"🧠 AI Analysis complete: Severity={analysis.get('severity')}/10, Risk={analysis.get('risk_level')}/10")
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI analysis JSON: {e}")
                logger.debug(f"Raw response: {response.text}")
                return None

        except Exception as e:
            logger.error(f"AI health analysis failed: {e}")
            return None

    async def _execute_recovery(self, action: Dict[str, Any], health_event: Dict[str, Any]):
        """
        Execute a recovery action recommended by AI analysis.

        Args:
            action: Recovery action details
            health_event: Original health event
        """
        try:
            action_name = action.get('name', 'unknown')
            parameters = action.get('parameters', {})

            logger.info(f"🔧 Executing recovery action: {action_name}")

            # Get recovery manager from services
            recovery_manager = None
            if hasattr(self, 'recovery_manager') and self.recovery_manager:
                recovery_manager = self.recovery_manager
            else:
                logger.warning("No recovery manager available")
                return

            # Add health event context to parameters
            parameters['health_event'] = health_event
            parameters['component'] = health_event.get('component')

            # Map AI action names to recovery manager actions
            action_map = {
                'reconnect_database': 'reconnect_database',
                'restart_component': 'restart_component',
                'clear_cache': 'clear_component_cache',
                'reset_state': 'reset_component_state'
            }

            recovery_action = action_map.get(action_name, action_name)

            # Execute recovery
            success = await recovery_manager.execute_recovery_action(
                component=health_event.get('component', 'unknown'),
                action=recovery_action,
                parameters=parameters
            )

            if success:
                logger.info(f"✅ Recovery action '{action_name}' completed successfully")

                # Verify recovery after a delay
                await asyncio.sleep(5)
                await self._verify_recovery(health_event.get('component'))
            else:
                logger.error(f"❌ Recovery action '{action_name}' failed")

        except Exception as e:
            logger.error(f"Error executing recovery action: {e}")
            import traceback
            traceback.print_exc()

    async def _verify_recovery(self, component: str):
        """
        Verify that recovery was successful by checking component health.

        Args:
            component: Component name to verify
        """
        try:
            if not self.health_monitor:
                logger.warning("No health monitor available for verification")
                return

            # Re-check component health
            health_status = await self.health_monitor.check_component_health(component)

            if health_status == "healthy":
                logger.info(f"✅ Recovery verified: {component} is now healthy")
            elif health_status == "degraded":
                logger.warning(f"⚠️ Partial recovery: {component} is degraded")
            else:
                logger.error(f"❌ Recovery failed: {component} is still {health_status}")

        except Exception as e:
            logger.error(f"Error verifying recovery: {e}")


# Convenience function for external use
async def create_autonomous_system(config: Optional[Dict[str, Any]] = None, torin_brain=None) -> AutonomousCoordinator:
    """Create and initialize an autonomous system coordinator"""
    coordinator = AutonomousCoordinator(config, torin_brain=torin_brain)
    if await coordinator.initialize():
        return coordinator
    else:
        raise RuntimeError("Failed to initialize autonomous system")


# Singleton instance
_autonomous_coordinator = None

async def get_autonomous_coordinator(config: Optional[Dict[str, Any]] = None, torin_brain=None) -> AutonomousCoordinator:
    """Get global autonomous coordinator instance (singleton)"""
    global _autonomous_coordinator
    if _autonomous_coordinator is None:
        _autonomous_coordinator = await create_autonomous_system(config, torin_brain=torin_brain)
    return _autonomous_coordinator
