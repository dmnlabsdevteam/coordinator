#!/usr/bin/env python3
"""
Shared types and enums for the autonomous system
Essential data structures without over-engineering
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime


# ============================================================================
# CORE ENUMS (Simplified from 26+ enums to essential ones)
# ============================================================================

class SystemMode(Enum):
    """System operational modes"""
    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class TaskType(Enum):
    """Task type classification"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EXECUTION = "execution"
    PLANNING = "planning"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    OPTIMIZATION = "optimization"
    SECURITY_REMEDIATION = "security_remediation"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class Priority(Enum):
    """Priority levels for tasks and goals"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskSource(Enum):
    """Task source types for governance differentiation"""
    EXTRINSIC_JSON = "extrinsic_json"  # User-defined tasks from JSON
    API = "api"  # Tasks from API requests
    MANUAL = "manual"  # Manually created by human
    AUTONOMOUS = "autonomous"  # AI-generated tasks
    SYSTEM = "system"  # System-generated tasks
    SECURITY_AUDIT = "security_audit"  # Security audit worker findings


# ============================================================================
# CORE DATA STRUCTURES (Simplified from 40+ classes)
# ============================================================================

@dataclass
class Task:
    """Basic task representation"""
    id: str
    type: TaskType
    description: str
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    estimated_duration: float = 0.0  # minutes
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None

    # Phase 5A: Task source tracking for governance
    source: TaskSource = TaskSource.AUTONOMOUS
    created_by: str = "autonomous_coordinator"
    governance_approved: bool = False
    governance_action_id: Optional[str] = None

    # Task execution tracking
    success_criteria: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and fix types after initialization"""
        # Ensure priority is a Priority enum, not a dict or other type
        if not isinstance(self.priority, Priority):
            if isinstance(self.priority, dict):
                # If it's a dict, try to extract the value or use default
                priority_value = self.priority.get('value', 'medium') if 'value' in self.priority else 'medium'
                self.priority = Priority[priority_value.upper()] if isinstance(priority_value, str) else Priority.MEDIUM
            elif isinstance(self.priority, str):
                # Convert string to Priority enum
                try:
                    self.priority = Priority[self.priority.upper()]
                except (KeyError, AttributeError):
                    self.priority = Priority.MEDIUM
            else:
                # Default fallback
                self.priority = Priority.MEDIUM


@dataclass
class Goal:
    """Simplified autonomous goal"""
    id: str
    description: str
    priority: Priority = Priority.MEDIUM
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "active"
    # Intrinsic motivation fields
    expected_novelty: float = 0.5  # Expected novelty from pursuing this goal
    expected_competence_gain: float = 0.5  # Expected skill improvement
    curiosity_value: float = 0.5  # Curiosity-driven interest in this goal
    intrinsic_reward_potential: float = 0.5  # Overall intrinsic reward potential
    
    def __post_init__(self):
        """Validate and fix types after initialization"""
        # Ensure all intrinsic values are floats, not dicts
        for field_name in ['expected_novelty', 'expected_competence_gain', 'curiosity_value', 'intrinsic_reward_potential']:
            value = getattr(self, field_name)
            if isinstance(value, dict):
                # If it's a dict, try to extract the value or use default
                value = value.get('value', 0.5) if 'value' in value else 0.5
            # Ensure it's a float
            try:
                setattr(self, field_name, float(value))
            except (TypeError, ValueError):
                setattr(self, field_name, 0.5)


@dataclass
class PerceptionData:
    """Simplified perception information"""
    source: str
    data_type: str
    content: Dict[str, Any]
    confidence: float = 1.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """Simplified execution plan"""
    id: str
    goal_id: str
    tasks: List[Task]
    estimated_duration: float = 0.0
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """Current system state snapshot"""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    mode: SystemMode = SystemMode.AUTONOMOUS
    active_tasks: List[str] = field(default_factory=list)
    active_goals: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    resource_usage: float = 0.0


@dataclass
class LearningData:
    """Learning experience data for pattern recognition and adaptation"""
    context: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    success: bool
    confidence: float = 0.5
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)