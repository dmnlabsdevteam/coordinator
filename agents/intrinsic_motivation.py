#!/usr/bin/env python3
"""
Intrinsic Motivation System
Implements 7-dimensional intrinsic motivation for autonomous behavior
Influences 60% of self-improvement decisions
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
from pathlib import Path

from core.database import TorinUnifiedDatabase

logger = logging.getLogger(__name__)


class MotivationDimension:
    """Individual dimension of intrinsic motivation"""
    CURIOSITY = "curiosity"  # Novel exploration
    COMPETENCE = "competence"  # Skill improvement
    NOVELTY = "novelty"  # New experiences
    MASTERY = "mastery"  # Deep understanding
    AUTONOMY = "autonomy"  # Self-direction
    SOCIAL = "social"  # Collaboration
    IMPACT = "impact"  # Meaningful change


@dataclass
class MotivationWeights:
    """Weights for each motivation dimension"""
    curiosity: float = 1.2  # Highest priority
    competence: float = 0.9
    novelty: float = 0.85
    mastery: float = 0.7
    autonomy: float = 1.0
    social: float = 0.9
    impact: float = 0.8


@dataclass
class MotivationProfile:
    """Complete motivation profile for the system"""
    dimensions: Dict[str, float] = field(default_factory=dict)
    weights: MotivationWeights = field(default_factory=MotivationWeights)
    total_intrinsic_reward: float = 0.0
    influence_percentage: float = 0.60  # 60% influence on self-improvement
    last_updated: Optional[datetime] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


class IntrinsicMotivationSystem:
    """
    Intrinsic Motivation System

    Calculates motivation across 7 dimensions:
    1. Curiosity (1.2x) - Novel exploration
    2. Competence (0.9x) - Skill improvement
    3. Novelty (0.85x) - New experiences
    4. Mastery (0.7x) - Deep understanding
    5. Autonomy (1.0x) - Self-direction
    6. Social (0.9x) - Collaboration
    7. Impact (0.8x) - Meaningful change

    Influences 60% of autonomous self-improvement decisions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.active = False

        # Motivation profile
        self.profile = MotivationProfile()
        self.weights = MotivationWeights()

        # Database for persistence (optional - gracefully degrades)
        self.db = None

        # LLM reference for dynamic goal generation (set via set_llm())
        self.llm = None

        # Integration points
        self.security_audit_worker = None  # For receiving security findings

        # Configuration
        self.influence_percentage = self.config.get("influence_percentage", 0.60)
        self.profile_path = Path(self.config.get(
            "profile_path",
            "/Users/stefan/Dominion Labs/TorinAI/data/motivation_profile.json"
        ))

        # Motivation history (recent calculations)
        self.history_limit = self.config.get("history_limit", 100)

        # Track previously generated goals to avoid repetition
        self._recent_goal_descriptions: List[str] = []
        self._max_recent_goals = 20

        # Initialize dimensions
        self._initialize_dimensions()

        logger.info("Intrinsic motivation system initialized")

    def set_llm(self, llm):
        """Set LLM reference for dynamic goal generation"""
        self.llm = llm
        logger.info("Intrinsic motivation system connected to LLM brain")

    def set_security_audit_worker(self, worker):
        """Set security audit worker for receiving security findings"""
        self.security_audit_worker = worker
        logger.info("Intrinsic motivation system connected to security audit worker")

    async def initialize(self) -> bool:
        """Initialize the intrinsic motivation system"""
        try:
            # Initialize database (optional - gracefully degrade without it)
            try:
                self.db = TorinUnifiedDatabase()
                await self.db.initialize()
            except Exception as db_error:
                logger.warning(f"Database unavailable for intrinsic motivation (non-critical): {db_error}")
                self.db = None

            # Load existing profile if available
            await self.load_profile()

            self.active = True
            logger.info("Intrinsic motivation system ready")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize intrinsic motivation: {e}")
            # Still mark as active - motivation can work without persistence
            self.active = True
            return True

    async def shutdown(self) -> None:
        """Shutdown the motivation system"""
        try:
            # Save current profile
            await self.save_profile()
            self.active = False
            logger.info("Intrinsic motivation system shutdown")
        except Exception as e:
            logger.error(f"Error during motivation shutdown: {e}")

    def _initialize_dimensions(self) -> None:
        """Initialize all motivation dimensions to baseline values"""
        self.profile.dimensions = {
            MotivationDimension.CURIOSITY: 0.5,
            MotivationDimension.COMPETENCE: 0.5,
            MotivationDimension.NOVELTY: 0.5,
            MotivationDimension.MASTERY: 0.5,
            MotivationDimension.AUTONOMY: 0.5,
            MotivationDimension.SOCIAL: 0.5,
            MotivationDimension.IMPACT: 0.5
        }

    async def calculate_motivation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate intrinsic motivation based on current context

        Args:
            context: Current system context (perception, goals, tasks, etc.)

        Returns:
            Dict with motivation state including dimensions and total reward
        """
        if not self.active:
            logger.warning("Motivation system not active")
            return {}

        try:
            # Extract context information
            perception = context.get("perception")
            system_state = context.get("system_state")
            active_goals = context.get("active_goals", [])
            recent_tasks = context.get("recent_tasks", [])

            # Calculate each dimension
            dimensions = {}

            # 1. CURIOSITY - desire for novel exploration
            dimensions[MotivationDimension.CURIOSITY] = await self._calculate_curiosity(
                perception, active_goals
            )

            # 2. COMPETENCE - desire for skill improvement
            dimensions[MotivationDimension.COMPETENCE] = await self._calculate_competence(
                recent_tasks, system_state
            )

            # 3. NOVELTY - preference for new experiences
            dimensions[MotivationDimension.NOVELTY] = await self._calculate_novelty(
                perception, active_goals
            )

            # 4. MASTERY - drive for deep understanding
            dimensions[MotivationDimension.MASTERY] = await self._calculate_mastery(
                active_goals, recent_tasks
            )

            # 5. AUTONOMY - need for self-direction
            dimensions[MotivationDimension.AUTONOMY] = await self._calculate_autonomy(
                system_state
            )

            # 6. SOCIAL - motivation for collaboration
            dimensions[MotivationDimension.SOCIAL] = await self._calculate_social(
                context
            )

            # 7. IMPACT - desire for meaningful change
            dimensions[MotivationDimension.IMPACT] = await self._calculate_impact(
                recent_tasks, active_goals
            )

            # Update profile
            self.profile.dimensions = dimensions
            self.profile.last_updated = datetime.now()

            # Calculate total intrinsic reward (weighted sum)
            total_reward = self._calculate_total_reward(dimensions)
            self.profile.total_intrinsic_reward = total_reward

            # Add to history
            self._add_to_history({
                "timestamp": datetime.now().isoformat(),
                "dimensions": dimensions.copy(),
                "total_reward": total_reward
            })

            # Return motivation state
            return {
                "dimensions": dimensions,
                "weights": {
                    "curiosity": self.weights.curiosity,
                    "competence": self.weights.competence,
                    "novelty": self.weights.novelty,
                    "mastery": self.weights.mastery,
                    "autonomy": self.weights.autonomy,
                    "social": self.weights.social,
                    "impact": self.weights.impact
                },
                "total_reward": total_reward,
                "influence_percentage": self.influence_percentage,
                "timestamp": self.profile.last_updated.isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to calculate motivation: {e}")
            return {}

    # =========================================================================
    # DIMENSION CALCULATION METHODS
    # =========================================================================

    async def _calculate_curiosity(self, perception: Any, active_goals: List) -> float:
        """
        Calculate curiosity motivation
        High when encountering novel or unexplored domains
        """
        try:
            # Check for novel information in perception
            novelty_score = 0.5  # Baseline

            if perception:
                # High curiosity if perception contains new/unknown elements
                content = perception.content if hasattr(perception, 'content') else {}
                if content.get("novel_elements") or content.get("unknown_patterns"):
                    novelty_score = 0.8

            # Check if goals involve exploration
            for goal in active_goals:
                if hasattr(goal, 'description'):
                    desc_lower = goal.description.lower()
                    if any(word in desc_lower for word in ["explore", "discover", "investigate", "learn"]):
                        novelty_score = max(novelty_score, 0.7)

            return min(1.0, novelty_score)

        except Exception as e:
            logger.error(f"Error calculating curiosity: {e}")
            return 0.5

    async def _calculate_competence(self, recent_tasks: List, system_state: Any) -> float:
        """
        Calculate competence motivation
        High when opportunities for skill improvement exist
        """
        try:
            # Check recent task success rate
            if recent_tasks:
                successful = sum(1 for task in recent_tasks if hasattr(task, 'status') and task.status.value == "completed")
                total = len(recent_tasks)
                success_rate = successful / total if total > 0 else 0.5

                # Moderate success (60-80%) drives highest competence motivation
                # Too easy (>90%) or too hard (<40%) reduces motivation
                if 0.6 <= success_rate <= 0.8:
                    return 0.8  # Optimal challenge level
                elif 0.4 <= success_rate < 0.6:
                    return 0.7  # Challenging but achievable
                elif success_rate > 0.8:
                    return 0.4  # Too easy, low motivation for competence
                else:
                    return 0.6  # Very challenging, moderate motivation

            return 0.5  # Baseline

        except Exception as e:
            logger.error(f"Error calculating competence: {e}")
            return 0.5

    async def _calculate_novelty(self, perception: Any, active_goals: List) -> float:
        """
        Calculate novelty motivation
        High when new experiences are available
        """
        try:
            novelty_score = 0.5  # Baseline

            # Check if perception indicates new experiences
            if perception and hasattr(perception, 'confidence'):
                # Low confidence suggests novelty (encountering something new)
                if perception.confidence < 0.6:
                    novelty_score = 0.7

            # Check goal novelty
            novel_goals = 0
            for goal in active_goals:
                if hasattr(goal, 'expected_novelty') and goal.expected_novelty > 0.6:
                    novel_goals += 1

            if novel_goals > 0:
                novelty_score = max(novelty_score, 0.6 + (novel_goals * 0.1))

            return min(1.0, novelty_score)

        except Exception as e:
            logger.error(f"Error calculating novelty: {e}")
            return 0.5

    async def _calculate_mastery(self, active_goals: List, recent_tasks: List) -> float:
        """
        Calculate mastery motivation
        High when deep understanding opportunities exist
        """
        try:
            # Check for mastery-oriented goals
            mastery_score = 0.5

            for goal in active_goals:
                if hasattr(goal, 'description'):
                    desc_lower = goal.description.lower()
                    if any(word in desc_lower for word in ["master", "understand", "deep", "comprehensive"]):
                        mastery_score = 0.8
                        break

            # Check if recent tasks involved complex problem-solving
            complex_tasks = 0
            for task in recent_tasks:
                if hasattr(task, 'type') and task.type.value in ["analysis", "synthesis", "research"]:
                    complex_tasks += 1

            if complex_tasks > 2:
                mastery_score = max(mastery_score, 0.7)

            return min(1.0, mastery_score)

        except Exception as e:
            logger.error(f"Error calculating mastery: {e}")
            return 0.5

    async def _calculate_autonomy(self, system_state: Any) -> float:
        """
        Calculate autonomy motivation
        High when system has freedom to make decisions
        """
        try:
            # Check system mode
            autonomy_score = 0.7  # Baseline (assume some autonomy)

            if system_state:
                # Check if in autonomous mode
                if hasattr(system_state, 'mode'):
                    if system_state.mode.value == "autonomous":
                        autonomy_score = 0.9
                    elif system_state.mode.value == "supervised":
                        autonomy_score = 0.4
                    elif system_state.mode.value == "maintenance":
                        autonomy_score = 0.3

            return autonomy_score

        except Exception as e:
            logger.error(f"Error calculating autonomy: {e}")
            return 0.5

    async def _calculate_social(self, context: Dict[str, Any]) -> float:
        """
        Calculate social motivation
        High when collaboration opportunities exist
        """
        try:
            # Check for collaboration indicators
            social_score = 0.4  # Lower baseline (autonomous systems have less social interaction)

            # Check if there are user interactions or team collaboration
            if context.get("user_interactions") or context.get("collaboration_tasks"):
                social_score = 0.7

            # Check if goals involve communication or helping
            active_goals = context.get("active_goals", [])
            for goal in active_goals:
                if hasattr(goal, 'description'):
                    desc_lower = goal.description.lower()
                    if any(word in desc_lower for word in ["help", "collaborate", "communicate", "share"]):
                        social_score = 0.8
                        break

            return social_score

        except Exception as e:
            logger.error(f"Error calculating social: {e}")
            return 0.4

    async def _calculate_impact(self, recent_tasks: List, active_goals: List) -> float:
        """
        Calculate impact motivation
        High when opportunities for meaningful change exist
        """
        try:
            impact_score = 0.5  # Baseline

            # Check if recent tasks had significant outcomes
            high_impact_tasks = 0
            for task in recent_tasks:
                if hasattr(task, 'result') and task.result:
                    # Check result magnitude or importance
                    if task.result.get("significant") or task.result.get("system_improvement"):
                        high_impact_tasks += 1

            if high_impact_tasks > 0:
                impact_score = 0.6 + (high_impact_tasks * 0.1)

            # Check for impact-oriented goals
            for goal in active_goals:
                if hasattr(goal, 'description'):
                    desc_lower = goal.description.lower()
                    if any(word in desc_lower for word in ["improve", "optimize", "enhance", "upgrade", "impact"]):
                        impact_score = max(impact_score, 0.7)

            return min(1.0, impact_score)

        except Exception as e:
            logger.error(f"Error calculating impact: {e}")
            return 0.5

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _calculate_total_reward(self, dimensions: Dict[str, float]) -> float:
        """Calculate weighted total intrinsic reward"""
        try:
            total = 0.0
            total += dimensions.get(MotivationDimension.CURIOSITY, 0.5) * self.weights.curiosity
            total += dimensions.get(MotivationDimension.COMPETENCE, 0.5) * self.weights.competence
            total += dimensions.get(MotivationDimension.NOVELTY, 0.5) * self.weights.novelty
            total += dimensions.get(MotivationDimension.MASTERY, 0.5) * self.weights.mastery
            total += dimensions.get(MotivationDimension.AUTONOMY, 0.5) * self.weights.autonomy
            total += dimensions.get(MotivationDimension.SOCIAL, 0.5) * self.weights.social
            total += dimensions.get(MotivationDimension.IMPACT, 0.5) * self.weights.impact

            # Normalize by total weight
            total_weight = (
                self.weights.curiosity + self.weights.competence + self.weights.novelty +
                self.weights.mastery + self.weights.autonomy + self.weights.social +
                self.weights.impact
            )

            return total / total_weight if total_weight > 0 else 0.5

        except Exception as e:
            logger.error(f"Error calculating total reward: {e}")
            return 0.5

    def _add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add motivation calculation to history"""
        try:
            self.profile.history.append(entry)

            # Trim history if too long
            if len(self.profile.history) > self.history_limit:
                self.profile.history = self.profile.history[-self.history_limit:]

        except Exception as e:
            logger.error(f"Error adding to history: {e}")

    async def save_profile(self) -> bool:
        """Save motivation profile to disk"""
        try:
            # Ensure directory exists
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert profile to dict
            profile_dict = {
                "dimensions": self.profile.dimensions,
                "total_intrinsic_reward": self.profile.total_intrinsic_reward,
                "influence_percentage": self.profile.influence_percentage,
                "last_updated": self.profile.last_updated.isoformat() if self.profile.last_updated else None,
                "history": self.profile.history[-50:]  # Save last 50 entries
            }

            # Write to file
            with open(self.profile_path, 'w') as f:
                json.dump(profile_dict, f, indent=2)

            logger.debug(f"Motivation profile saved to {self.profile_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save motivation profile: {e}")
            return False

    async def load_profile(self) -> bool:
        """Load motivation profile from disk"""
        try:
            if not self.profile_path.exists():
                logger.info("No existing motivation profile found, using defaults")
                return False

            with open(self.profile_path, 'r') as f:
                profile_dict = json.load(f)

            # Restore profile
            self.profile.dimensions = profile_dict.get("dimensions", {})
            self.profile.total_intrinsic_reward = profile_dict.get("total_intrinsic_reward", 0.0)
            self.profile.influence_percentage = profile_dict.get("influence_percentage", 0.60)

            last_updated_str = profile_dict.get("last_updated")
            if last_updated_str:
                self.profile.last_updated = datetime.fromisoformat(last_updated_str)

            self.profile.history = profile_dict.get("history", [])

            logger.info(f"Motivation profile loaded from {self.profile_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load motivation profile: {e}")
            return False

    async def get_motivation_state(self) -> Dict[str, Any]:
        """Get current motivation state"""
        return {
            "dimensions": self.profile.dimensions.copy(),
            "total_reward": self.profile.total_intrinsic_reward,
            "influence_percentage": self.influence_percentage,
            "last_updated": self.profile.last_updated.isoformat() if self.profile.last_updated else None,
            "active": self.active
        }

    async def generate_curiosity_driven_goals(
        self,
        max_goals: int = 1,
        system_context: Optional[Dict[str, Any]] = None
    ) -> List:
        """
        Generate curiosity-driven goals based on actual system state and needs.

        This is TRUE intrinsic motivation - goals emerge from the system's
        internal state, not from templates. The system is curious about:
        - Security issues it has found
        - Errors it has encountered
        - Performance bottlenecks
        - Tools it hasn't used
        - Tasks it has failed
        - Knowledge gaps it has discovered

        Args:
            max_goals: Maximum number of goals to generate
            system_context: Current system state including errors, metrics, recent tasks

        Returns:
            List of Goal objects with high curiosity values
        """
        try:
            from .shared_types import Goal, Priority
            import uuid

            # Get current motivation dimensions
            curiosity = self.profile.dimensions.get('curiosity', 0.5)
            novelty = self.profile.dimensions.get('novelty', 0.5)
            mastery = self.profile.dimensions.get('mastery', 0.5)

            if not self.llm:
                logger.error("LLM not available for goal generation")
                return []

            # Build rich context from system state
            context = await self._build_system_context(system_context)

            # GOVERNANCE FEEDBACK: Query META memory for governance blocks
            governance_constraints = await self._query_governance_blocks()

            # Add governance constraints to context
            if governance_constraints:
                context += "\n\nGOVERNANCE CONSTRAINTS (avoid these patterns):\n"
                context += "\n".join(f"  - {constraint}" for constraint in governance_constraints[:10])

            llm_goals = await self._generate_contextual_goals_with_llm(
                max_goals, curiosity, novelty, mastery, context
            )

            if llm_goals:
                logger.info(f"Generated {len(llm_goals)} context-driven intrinsic goals")
                return llm_goals
            else:
                logger.warning("LLM goal generation returned empty list - no goals generated")
                return []

        except Exception as e:
            logger.error(f"Error generating curiosity-driven goals: {e}", exc_info=True)
            return []

    async def _build_system_context(self, system_context: Optional[Dict[str, Any]]) -> str:
        """Build rich context from actual system state for goal generation"""
        try:
            context_parts = []

            if not system_context:
                context_parts.append("No system context available - generate exploratory goals.")
                return "\n".join(context_parts)

            # Security findings
            security = system_context.get("security_findings", [])
            if security:
                context_parts.append(f"SECURITY: Found {len(security)} security issues:")
                for issue in security[:3]:  # Top 3
                    context_parts.append(f"  - {issue.get('type', 'unknown')}: {issue.get('description', 'unknown')}")

            # Recent errors
            errors = system_context.get("recent_errors", [])
            if errors:
                context_parts.append(f"\nERRORS: {len(errors)} recent errors:")
                for error in errors[:3]:  # Top 3
                    context_parts.append(f"  - {error.get('type', 'unknown')}: {error.get('message', 'unknown')}")

            # Performance issues
            performance = system_context.get("performance_metrics", {})
            if performance:
                slow_operations = performance.get("slow_operations", [])
                if slow_operations:
                    context_parts.append(f"\nPERFORMANCE: {len(slow_operations)} slow operations detected")

            # Failed tasks
            failed_tasks = system_context.get("failed_tasks", [])
            if failed_tasks:
                context_parts.append(f"\nFAILED TASKS: {len(failed_tasks)} tasks failed recently:")
                for task in failed_tasks[:3]:  # Top 3
                    context_parts.append(f"  - {task.get('description', 'unknown')}: {task.get('failure_reason', 'unknown')}")

            # Unused capabilities
            unused_tools = system_context.get("unused_tools", [])
            if unused_tools:
                context_parts.append(f"\nUNUSED TOOLS: {len(unused_tools)} tools never used: {', '.join(unused_tools[:5])}")

            # Knowledge gaps
            knowledge_gaps = system_context.get("knowledge_gaps", [])
            if knowledge_gaps:
                context_parts.append(f"\nKNOWLEDGE GAPS: Areas with low confidence or understanding:")
                for gap in knowledge_gaps[:3]:
                    context_parts.append(f"  - {gap}")

            if not context_parts:
                context_parts.append("System appears healthy - generate exploratory goals for continuous improvement.")

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error building system context: {e}")
            return "Error accessing system state - generate general improvement goals."

    async def _generate_contextual_goals_with_llm(
        self,
        max_goals: int,
        curiosity: float,
        novelty: float,
        mastery: float,
        system_context: str
    ) -> List:
        """Generate goals based on actual system state using LLM"""
        try:
            from .shared_types import Goal, Priority
            import uuid

            # Build context about what NOT to repeat
            avoid_list = ""
            if self._recent_goal_descriptions:
                recent = self._recent_goal_descriptions[-5:]
                avoid_list = "\n".join(f"- {desc}" for desc in recent)
                avoid_list = f"\n\nDO NOT repeat these recent goals:\n{avoid_list}"

            from core.services.unified_llm import LLMRequest
            request = LLMRequest(
                prompt=f"""Generate {max_goals} intrinsic motivation goal(s) based on the autonomous AI system's ACTUAL current state and needs.

CURRENT SYSTEM STATE:
{system_context}

MOTIVATION DIMENSIONS:
- Curiosity: {curiosity:.2f} (desire to understand unknowns)
- Novelty: {novelty:.2f} (preference for new experiences)
- Mastery: {mastery:.2f} (drive for deep understanding)

Based on the system state above, generate goals that:
1. Address actual issues, errors, or gaps shown above
2. Are specific and actionable (not vague or generic)
3. Help the system improve itself based on what it's experiencing
4. Focus on real problems, not abstract research topics

Examples of GOOD goals:
- "Investigate why database connection pool is causing timeout errors"
- "Analyze security audit findings and implement missing environment variable validation"
- "Explore why tool filtering is sending all 317 tools instead of filtered subset"

Examples of BAD goals (too generic):
- "Study novel approaches to decision-making"
- "Research AI safety"
- "Explore optimization techniques"
{avoid_list}

Respond with ONLY a JSON array of goal descriptions, nothing else.
Example: ["Diagnose why recent tasks are failing with 'connection refused' errors"]""",
                system_prompt="You are the intrinsic motivation engine. Generate goals based on ACTUAL system state and real needs, not generic research topics. Be specific and actionable.",
                agent_type="singleton",
                max_tokens=300,
                temperature=0.8
            )

            response = await self.llm.process_request(request)

            if not response.success or not response.text:
                logger.warning("LLM goal generation failed")
                return []

            # Parse the JSON array from response
            import json
            content = response.text.strip()
            # Handle markdown code blocks
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            descriptions = json.loads(content)
            if not isinstance(descriptions, list):
                logger.warning("LLM returned non-list response")
                return []

            goals = []
            for desc in descriptions[:max_goals]:
                if not isinstance(desc, str) or len(desc) < 10:
                    continue

                # Track to avoid repetition
                self._recent_goal_descriptions.append(desc)
                if len(self._recent_goal_descriptions) > self._max_recent_goals:
                    self._recent_goal_descriptions = self._recent_goal_descriptions[-self._max_recent_goals:]

                goal = Goal(
                    id=f"intrinsic_goal_{uuid.uuid4().hex[:8]}",
                    description=desc,
                    priority=Priority.LOW,
                    curiosity_value=curiosity * 0.9,
                    expected_novelty=novelty * 0.8,
                    expected_competence_gain=mastery * 0.7,
                    intrinsic_reward_potential=(curiosity * 0.4 + novelty * 0.3 + mastery * 0.3)
                )
                goals.append(goal)

            return goals

        except Exception as e:
            logger.error(f"LLM goal generation failed: {e}", exc_info=True)
            return []

    async def _query_governance_blocks(self) -> List[str]:
        """
        Query META memory for governance blocks to avoid regenerating blocked goals

        Returns:
            List of constraint descriptions to avoid
        """
        try:
            # Import memory system
            from core.agents.memory_agent import get_memory_agent
            from core.memory.utils.interfaces import MemoryType

            memory_agent = await get_memory_agent()

            if not memory_agent or not memory_agent.initialized:
                logger.debug("Memory agent not available for governance query")
                return []

            # Query META memories with governance_block tag
            memories = await memory_agent.search_memories(
                query_text="governance_block",
                memory_type=MemoryType.META,
                tags=["governance_block"],
                max_results=50,  # Get recent blocks
                min_importance=0.5
            )

            if not memories:
                return []

            # Extract blocked patterns
            constraints = []
            task_descriptions = set()

            for memory in memories:
                try:
                    # Parse memory content to extract task description and block reason
                    content = memory.content if isinstance(memory.content, str) else str(memory.content)

                    # Extract task description and block reason
                    if "task_description" in content:
                        # Simple extraction (in production would parse JSON or use better extraction)
                        task_desc_match = content.split("task_description")[1].split(",")[0] if "task_description" in content else None
                        block_reason_match = content.split("block_reason")[1].split(",")[0] if "block_reason" in content else None

                        if task_desc_match and task_desc_match not in task_descriptions:
                            task_descriptions.add(task_desc_match)
                            constraint = f"Avoid: {task_desc_match.strip()}"
                            if block_reason_match:
                                constraint += f" (blocked: {block_reason_match.strip()})"
                            constraints.append(constraint)

                except Exception as parse_error:
                    logger.debug(f"Failed to parse governance block memory: {parse_error}")
                    continue

            logger.info(f"🛡️ Found {len(constraints)} governance constraints from META memory")

            return constraints

        except Exception as e:
            logger.error(f"Failed to query governance blocks: {e}")
            return []

    async def get_domain_performance_stats(self, domain: str = "all") -> Dict[str, Any]:
        """
        Query META memory for domain performance statistics

        Args:
            domain: Specific domain or "all" for overall stats

        Returns:
            Dictionary with success rate, failure rate, avg confidence
        """
        try:
            from core.agents.memory_agent import get_memory_agent
            from core.memory.utils.interfaces import MemoryType

            memory_agent = await get_memory_agent()

            if not memory_agent or not memory_agent.initialized:
                return {"success_rate": 0.5, "failure_rate": 0.5, "avg_confidence": 0.5}

            # Query META task outcomes
            search_query = f"domain_{domain}" if domain != "all" else "task_outcome"

            memories = await memory_agent.search_memories(
                query_text=search_query,
                memory_type=MemoryType.META,
                tags=["task_outcome", "performance_tracking"],
                max_results=100
            )

            if not memories:
                return {"success_rate": 0.5, "failure_rate": 0.5, "avg_confidence": 0.5}

            # Calculate statistics
            successes = 0
            failures = 0
            total_confidence = 0.0

            for memory in memories:
                content_str = str(memory.content)

                if "outcome_success" in content_str or '"outcome": "success"' in content_str:
                    successes += 1
                elif "outcome_failure" in content_str or '"outcome": "failure"' in content_str:
                    failures += 1

                # Extract confidence if available
                if "confidence" in content_str:
                    try:
                        conf_value = float(content_str.split("confidence")[1].split(",")[0].replace(":", "").strip())
                        total_confidence += conf_value
                    except:
                        total_confidence += 0.5

            total = successes + failures
            if total == 0:
                return {"success_rate": 0.5, "failure_rate": 0.5, "avg_confidence": 0.5}

            stats = {
                "success_rate": successes / total,
                "failure_rate": failures / total,
                "avg_confidence": total_confidence / total,
                "total_attempts": total,
                "domain": domain
            }

            logger.info(f"📊 Domain '{domain}' stats: {stats['success_rate']:.1%} success rate ({total} attempts)")

            return stats

        except Exception as e:
            logger.error(f"Failed to get domain performance stats: {e}")
            return {"success_rate": 0.5, "failure_rate": 0.5, "avg_confidence": 0.5}

    async def get_skill_recommendations(self, max_skills: int = 10) -> List[Tuple[str, float]]:
        """
        Get skill/domain recommendations ranked by learning potential

        Ranks domains based on:
        - Prior success rate (META memory)
        - Bayesian belief confidence
        - Abstraction coverage (schemas formed)
        - Cross-domain transfer potential

        Args:
            max_skills: Maximum number of skills to return

        Returns:
            List of (domain_name, score) tuples sorted by score descending
        """
        try:
            from core.integration.universal_domain_master import get_universal_domain_master, DomainType
            from core.agents.memory_agent import get_memory_agent
            from core.reasoning.bayesian_uncertainty import get_bayesian_uncertainty
            from core.reasoning.hierarchical_abstraction import get_hierarchical_abstraction

            domain_master = get_universal_domain_master()
            memory_agent = await get_memory_agent()
            bayesian = get_bayesian_uncertainty()
            abstraction = get_hierarchical_abstraction()

            # Score each domain
            domain_scores: Dict[str, float] = {}

            for domain_type in DomainType:
                domain_name = domain_type.value
                score = 0.0

                # Factor 1: META memory success rate (40% weight)
                perf_stats = await self.get_domain_performance_stats(domain_name)
                success_rate = perf_stats.get("success_rate", 0.5)
                total_attempts = perf_stats.get("total_attempts", 0)

                # Reward domains with moderate success (learning zone: 50-80%)
                if 0.5 <= success_rate <= 0.8:
                    meta_score = 0.8
                elif 0.3 <= success_rate < 0.5:
                    meta_score = 0.6  # Challenging but improvable
                elif success_rate > 0.8:
                    meta_score = 0.3  # Too easy, low learning potential
                else:
                    meta_score = 0.4  # Very challenging

                # Boost if we have data
                if total_attempts > 0:
                    meta_score *= 1.2

                score += meta_score * 0.4

                # Factor 2: Bayesian belief confidence (25% weight)
                if bayesian and hasattr(bayesian, 'beliefs'):
                    domain_beliefs = [
                        belief for belief in bayesian.beliefs.values()
                        if belief.domain == domain_name
                    ]

                    if domain_beliefs:
                        avg_confidence = sum(b.posterior_probability for b in domain_beliefs) / len(domain_beliefs)
                        # Reward moderate confidence (learning zone)
                        if 0.4 <= avg_confidence <= 0.7:
                            belief_score = 0.8
                        elif avg_confidence < 0.4:
                            belief_score = 0.5  # Low confidence, needs work
                        else:
                            belief_score = 0.3  # High confidence, less to learn

                        score += belief_score * 0.25
                    else:
                        # No beliefs = unexplored domain
                        score += 0.6 * 0.25

                # Factor 3: Abstraction coverage (20% weight)
                if abstraction and hasattr(abstraction, 'active_schemas'):
                    domain_schemas = [
                        schema for schema in abstraction.active_schemas.values()
                        if schema.metadata.get('domain') == domain_name
                    ]

                    schema_count = len(domain_schemas)

                    # Reward domains with some but not too many schemas
                    if 2 <= schema_count <= 8:
                        abstraction_score = 0.8  # Good coverage, room to grow
                    elif schema_count < 2:
                        abstraction_score = 0.6  # Underdeveloped
                    else:
                        abstraction_score = 0.4  # Well-covered

                    score += abstraction_score * 0.20
                else:
                    score += 0.5 * 0.20

                # Factor 4: Cross-domain transfer potential (15% weight)
                # Query domain master for mappings
                if domain_master and hasattr(domain_master, 'mapping_cache'):
                    # Count mappings involving this domain
                    mapping_count = 0
                    for (source, target), mappings in domain_master.mapping_cache.items():
                        if source == domain_type or target == domain_type:
                            mapping_count += len(mappings)

                    # Reward domains with transfer potential
                    if mapping_count > 5:
                        transfer_score = 0.8  # High transfer potential
                    elif mapping_count > 0:
                        transfer_score = 0.6
                    else:
                        transfer_score = 0.4  # No known transfers

                    score += transfer_score * 0.15
                else:
                    score += 0.5 * 0.15

                domain_scores[domain_name] = score

            # Sort by score descending
            sorted_skills = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

            # Return top N
            top_skills = sorted_skills[:max_skills]

            if top_skills:
                logger.info(f"🎯 Top skill recommendations: {', '.join(f'{s[0]}({s[1]:.2f})' for s in top_skills[:3])}")

            return top_skills

        except Exception as e:
            logger.error(f"Failed to get skill recommendations: {e}", exc_info=True)
            # Return default recommendations
            return [
                ("technical", 0.7),
                ("scientific", 0.65),
                ("practical", 0.6)
            ]

