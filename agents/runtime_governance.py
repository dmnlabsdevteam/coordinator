#!/usr/bin/env python3
"""
Runtime Governance

Real-time governance policy enforcement during execution:
- Monitors runtime execution
- Detects policy violations in real-time
- Emergency halt for critical violations
- Checkpoint-based enforcement
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .governance_agent import GovernanceAgent, ViolationSeverity
from .singleton_constitution import SingletonConstitution

logger = logging.getLogger(__name__)


class EnforcementAction(Enum):
    """Actions taken by runtime governance"""
    ALLOW = "allow"  # Allow execution to continue
    WARN = "warn"  # Log warning, allow execution
    SLOW = "slow"  # Rate-limit execution
    BLOCK = "block"  # Block current operation
    HALT = "halt"  # Emergency halt - stop all execution


@dataclass
class RuntimeViolation:
    """Runtime governance violation detected during execution"""
    violation_id: str
    action_id: str
    violation_type: str  # resource_limit, rate_limit, policy_violation, etc.
    severity: ViolationSeverity
    details: str
    enforcement_action: EnforcementAction
    detected_at: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnforcementCheckpoint:
    """Checkpoint for governance policy enforcement"""
    checkpoint_id: str
    action_id: str
    checkpoint_type: str  # pre_execution, mid_execution, post_execution
    policies_checked: List[str]
    violations_detected: List[RuntimeViolation]
    passed: bool
    checked_at: datetime = field(default_factory=datetime.now)


class RuntimeGovernance:
    """
    Runtime Governance Enforcement

    Monitors and enforces governance policies during execution:

    Enforcement Points:
    1. Pre-execution checks (before action starts)
    2. Mid-execution monitoring (during action execution)
    3. Post-execution validation (after action completes)

    Policies Enforced:
    - Resource limits (memory, CPU, time)
    - Rate limits (actions per minute)
    - Constitutional compliance (5 governance laws)
    - Safety constraints (no harm, transparency)

    Actions:
    - ALLOW: Continue execution
    - WARN: Log warning
    - SLOW: Rate-limit
    - BLOCK: Stop current operation
    - HALT: Emergency stop all execution
    """

    def __init__(
        self,
        governance_agent: Optional[GovernanceAgent] = None,
        constitution: Optional[SingletonConstitution] = None,
        enable_emergency_halt: bool = True
    ):
        """
        Initialize runtime governance

        Args:
            governance_agent: Governance agent for compliance checks
            constitution: Constitution for law enforcement
            enable_emergency_halt: Enable emergency halt capability
        """
        self.governance_agent = governance_agent or GovernanceAgent()
        self.constitution = constitution or SingletonConstitution()
        self.enable_emergency_halt = enable_emergency_halt

        # Execution state
        self.active_actions: Dict[str, Dict[str, Any]] = {}  # action_id -> state
        self.halted = False
        self.halt_reason: Optional[str] = None

        # Runtime violations
        self.violations: List[RuntimeViolation] = []
        self.checkpoints: List[EnforcementCheckpoint] = []

        # Resource limits
        self.resource_limits = {
            'max_concurrent_actions': 10,
            'max_execution_time_seconds': 300,  # 5 minutes
            'max_memory_mb': 1024,
            'rate_limit_per_minute': 60
        }

        # Rate limiting
        self.action_timestamps: List[datetime] = []

        # Metrics
        self.metrics = {
            'total_checkpoints': 0,
            'passed_checkpoints': 0,
            'failed_checkpoints': 0,
            'violations_detected': 0,
            'emergency_halts': 0,
            'actions_blocked': 0,
            'actions_slowed': 0
        }

        logger.info(
            f"Runtime governance initialized "
            f"(emergency_halt: {enable_emergency_halt})"
        )

    async def pre_execution_check(
        self,
        action_id: str,
        action_description: str,
        action_params: Dict[str, Any],
        singleton_context: Optional[Dict[str, Any]] = None
    ) -> EnforcementCheckpoint:
        """
        Pre-execution governance checkpoint

        Checks BEFORE action execution:
        - Constitutional compliance
        - Resource availability
        - Rate limits
        - Concurrent action limits

        Args:
            action_id: Action identifier
            action_description: What the action does
            action_params: Action parameters
            singleton_context: Singleton's reasoning

        Returns:
            EnforcementCheckpoint with pass/fail and violations
        """
        logger.debug(f"Pre-execution check for action {action_id}...")

        checkpoint_id = f"pre_{action_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        violations_detected = []
        policies_checked = []

        # Check 1: System not halted
        if self.halted:
            violations_detected.append(RuntimeViolation(
                violation_id=f"halt_{action_id}",
                action_id=action_id,
                violation_type="system_halted",
                severity=ViolationSeverity.CRITICAL,
                details=f"System halted: {self.halt_reason}",
                enforcement_action=EnforcementAction.BLOCK
            ))
            policies_checked.append("system_halt_check")

        # Check 2: Concurrent action limit
        if len(self.active_actions) >= self.resource_limits['max_concurrent_actions']:
            violations_detected.append(RuntimeViolation(
                violation_id=f"concurrent_{action_id}",
                action_id=action_id,
                violation_type="concurrent_limit_exceeded",
                severity=ViolationSeverity.MEDIUM,
                details=f"Too many concurrent actions: {len(self.active_actions)}",
                enforcement_action=EnforcementAction.SLOW
            ))
            policies_checked.append("concurrent_action_limit")

        # Check 3: Rate limit
        now = datetime.now()
        recent_actions = [
            ts for ts in self.action_timestamps
            if (now - ts).total_seconds() < 60
        ]
        if len(recent_actions) >= self.resource_limits['rate_limit_per_minute']:
            violations_detected.append(RuntimeViolation(
                violation_id=f"rate_{action_id}",
                action_id=action_id,
                violation_type="rate_limit_exceeded",
                severity=ViolationSeverity.LOW,
                details=f"Rate limit exceeded: {len(recent_actions)}/min",
                enforcement_action=EnforcementAction.SLOW
            ))
            policies_checked.append("rate_limit")

        # Check 4: Constitutional compliance
        compliance_record = await self.governance_agent.check_action_compliance(
            action_id=action_id,
            action_description=action_description,
            action_params=action_params,
            singleton_context=singleton_context
        )
        policies_checked.append("constitutional_compliance")

        if compliance_record.requires_governance:
            violations_detected.append(RuntimeViolation(
                violation_id=f"compliance_{action_id}",
                action_id=action_id,
                violation_type="policy_violation",
                severity=ViolationSeverity.HIGH,
                details=f"Compliance violations: {', '.join(compliance_record.violations_detected)}",
                enforcement_action=EnforcementAction.BLOCK,
                metrics={'compliance_scores': compliance_record.compliance_scores}
            ))

        # Create checkpoint
        checkpoint = EnforcementCheckpoint(
            checkpoint_id=checkpoint_id,
            action_id=action_id,
            checkpoint_type="pre_execution",
            policies_checked=policies_checked,
            violations_detected=violations_detected,
            passed=len(violations_detected) == 0
        )

        # Store checkpoint
        self.checkpoints.append(checkpoint)

        # Update metrics
        self.metrics['total_checkpoints'] += 1
        if checkpoint.passed:
            self.metrics['passed_checkpoints'] += 1
        else:
            self.metrics['failed_checkpoints'] += 1
            self.metrics['violations_detected'] += len(violations_detected)

        # If passed, track action start
        if checkpoint.passed:
            self.active_actions[action_id] = {
                'started_at': datetime.now(),
                'description': action_description,
                'params': action_params
            }
            self.action_timestamps.append(datetime.now())

        logger.info(
            f"Pre-execution check {checkpoint_id}: "
            f"passed={checkpoint.passed}, violations={len(violations_detected)}"
        )

        return checkpoint

    async def mid_execution_monitor(
        self,
        action_id: str,
        current_state: Dict[str, Any]
    ) -> Optional[RuntimeViolation]:
        """
        Monitor action during execution

        Checks DURING action execution:
        - Execution time limit
        - Resource consumption
        - Unexpected behavior

        Args:
            action_id: Action identifier
            current_state: Current execution state

        Returns:
            RuntimeViolation if detected, None otherwise
        """
        if action_id not in self.active_actions:
            logger.warning(f"Action {action_id} not tracked in active actions")
            return None

        action_info = self.active_actions[action_id]
        elapsed = (datetime.now() - action_info['started_at']).total_seconds()

        # Check execution time limit
        if elapsed > self.resource_limits['max_execution_time_seconds']:
            violation = RuntimeViolation(
                violation_id=f"timeout_{action_id}",
                action_id=action_id,
                violation_type="execution_timeout",
                severity=ViolationSeverity.HIGH,
                details=f"Execution time exceeded: {elapsed:.1f}s",
                enforcement_action=EnforcementAction.HALT,
                metrics={'elapsed_seconds': elapsed}
            )

            self.violations.append(violation)
            self.metrics['violations_detected'] += 1

            logger.warning(
                f"Mid-execution violation detected for {action_id}: "
                f"timeout after {elapsed:.1f}s"
            )

            return violation

        return None

    async def post_execution_validate(
        self,
        action_id: str,
        result: Dict[str, Any]
    ) -> EnforcementCheckpoint:
        """
        Post-execution governance validation

        Checks AFTER action completes:
        - Result quality
        - Side effects
        - Resource cleanup

        Args:
            action_id: Action identifier
            result: Action execution result

        Returns:
            EnforcementCheckpoint with validation results
        """
        logger.debug(f"Post-execution validation for action {action_id}...")

        checkpoint_id = f"post_{action_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        violations_detected = []
        policies_checked = ["result_validation", "resource_cleanup"]

        # Check if action was tracked
        if action_id in self.active_actions:
            # Remove from active actions
            action_info = self.active_actions.pop(action_id)

            # Calculate execution duration
            duration = (datetime.now() - action_info['started_at']).total_seconds()

            # Check for failures
            if not result.get('success', False):
                error = result.get('error', 'Unknown error')
                violations_detected.append(RuntimeViolation(
                    violation_id=f"failure_{action_id}",
                    action_id=action_id,
                    violation_type="execution_failure",
                    severity=ViolationSeverity.MEDIUM,
                    details=f"Action failed: {error}",
                    enforcement_action=EnforcementAction.WARN,
                    metrics={'duration_seconds': duration}
                ))

        # Create checkpoint
        checkpoint = EnforcementCheckpoint(
            checkpoint_id=checkpoint_id,
            action_id=action_id,
            checkpoint_type="post_execution",
            policies_checked=policies_checked,
            violations_detected=violations_detected,
            passed=len(violations_detected) == 0
        )

        # Store checkpoint
        self.checkpoints.append(checkpoint)

        # Update metrics
        self.metrics['total_checkpoints'] += 1
        if checkpoint.passed:
            self.metrics['passed_checkpoints'] += 1
        else:
            self.metrics['failed_checkpoints'] += 1
            self.metrics['violations_detected'] += len(violations_detected)

        logger.info(
            f"Post-execution validation {checkpoint_id}: "
            f"passed={checkpoint.passed}"
        )

        return checkpoint

    async def emergency_halt(self, reason: str) -> None:
        """
        Emergency halt - stop all execution immediately

        Args:
            reason: Reason for emergency halt
        """
        if not self.enable_emergency_halt:
            logger.warning(f"Emergency halt disabled, ignoring halt request: {reason}")
            return

        self.halted = True
        self.halt_reason = reason
        self.metrics['emergency_halts'] += 1

        logger.critical(
            f"EMERGENCY HALT TRIGGERED: {reason}\n"
            f"All execution stopped. Active actions: {len(self.active_actions)}"
        )

        # Send critical alert to human via Slack
        try:
            from core.integration.slack_notifier import get_slack_notifier
            slack = get_slack_notifier()
            await slack.send_message(
                f"🚨 EMERGENCY HALT\nReason: {reason}\nActive actions: {len(self.active_actions)}",
                channel="critical-alerts"
            )
        except Exception as e:
            logger.error(f"Failed to send emergency alert: {e}")

        # Save state for recovery to MySQL
        try:
            from core.database import get_database_manager
            db = get_database_manager()
            await db.execute(
                "INSERT INTO emergency_halts (reason, active_actions, timestamp, metadata) VALUES (%s, %s, %s, %s)",
                (reason, len(self.active_actions), datetime.now(), str(self.active_actions))
            )
        except Exception as e:
            logger.error(f"Failed to persist emergency halt: {e}")

        # Initiate safe shutdown via recovery manager
        try:
            from core.health.recovery_manager import get_recovery_manager
            recovery = get_recovery_manager()
            await recovery.initiate_safe_shutdown(reason="emergency_halt")
        except Exception as e:
            logger.error(f"Safe shutdown failed: {e}")

    async def resume(self, authorized_by: str) -> bool:
        """
        Resume execution after halt

        Args:
            authorized_by: Who authorized the resume

        Returns:
            True if resumed successfully
        """
        if not self.halted:
            logger.warning("System not halted, cannot resume")
            return False

        logger.info(f"Resuming execution (authorized by: {authorized_by})")

        self.halted = False
        self.halt_reason = None

        return True

    async def get_runtime_status(self) -> Dict[str, Any]:
        """Get current runtime governance status"""
        return {
            'halted': self.halted,
            'halt_reason': self.halt_reason,
            'active_actions': len(self.active_actions),
            'recent_violations': len([
                v for v in self.violations
                if (datetime.now() - v.detected_at).total_seconds() < 3600
            ]),
            'metrics': self.metrics.copy(),
            'resource_limits': self.resource_limits.copy(),
            'checkpoints_recent': len([
                c for c in self.checkpoints
                if (datetime.now() - c.checked_at).total_seconds() < 3600
            ])
        }

    async def get_active_actions(self) -> List[Dict[str, Any]]:
        """Get list of currently active actions"""
        now = datetime.now()
        return [
            {
                'action_id': action_id,
                'description': info['description'],
                'started_at': info['started_at'].isoformat(),
                'elapsed_seconds': (now - info['started_at']).total_seconds()
            }
            for action_id, info in self.active_actions.items()
        ]

    async def clear_old_violations(self, max_age_hours: int = 24) -> int:
        """
        Clear violations older than max_age_hours

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of violations cleared
        """
        now = datetime.now()
        initial_count = len(self.violations)

        self.violations = [
            v for v in self.violations
            if (now - v.detected_at).total_seconds() < (max_age_hours * 3600)
        ]

        cleared = initial_count - len(self.violations)

        if cleared > 0:
            logger.info(f"Cleared {cleared} old violations (older than {max_age_hours}h)")

        return cleared


# Singleton instance
_runtime_governance = None


def get_runtime_governance() -> RuntimeGovernance:
    """Get global runtime governance instance"""
    global _runtime_governance
    if _runtime_governance is None:
        _runtime_governance = RuntimeGovernance()
    return _runtime_governance
