# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for ParallelTaskOrchestrator.

Includes property-based tests for:
- Property 1: Max Parallel Limit Enforcement
- Property 7: Error Isolation
- Property 8: Failure Reporting Completeness
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from collections import Counter

from patient_agent_bench.runner.parallel import ParallelTaskOrchestrator, TaskResult
from patient_agent_bench.config import RolePoolManager
from patient_agent_bench.utils.retry import RetryConfig


# =============================================================================
# Unit Tests for TaskResult
# =============================================================================


class TestTaskResult:
    """Unit tests for TaskResult dataclass."""

    def test_successful_result(self):
        """Test TaskResult for successful task."""
        result = TaskResult(
            task_id="task-1",
            success=True,
            result={"data": "value"},
            duration=1.5,
        )
        assert result.task_id == "task-1"
        assert result.success is True
        assert result.result == {"data": "value"}
        assert result.error is None
        assert result.duration == 1.5

    def test_failed_result(self):
        """Test TaskResult for failed task."""
        result = TaskResult(
            task_id="task-2",
            success=False,
            error="Connection timeout",
            duration=0.5,
        )
        assert result.task_id == "task-2"
        assert result.success is False
        assert result.result is None
        assert result.error == "Connection timeout"
        assert result.duration == 0.5

    def test_default_values(self):
        """Test TaskResult default values."""
        result = TaskResult(task_id="task-3", success=True)
        assert result.result is None
        assert result.error is None
        assert result.duration == 0.0


# =============================================================================
# Unit Tests for ParallelTaskOrchestrator
# =============================================================================


class TestParallelTaskOrchestrator:
    """Unit tests for ParallelTaskOrchestrator class."""

    def test_default_initialization(self):
        """Test default initialization."""
        orchestrator = ParallelTaskOrchestrator()
        assert orchestrator.max_parallel == 1
        assert len(orchestrator.role_pool) == 0
        assert orchestrator.retry_config.max_retries == 5

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        role_pool = RolePoolManager(roles=["role1", "role2"])
        retry_config = RetryConfig(max_retries=3)
        orchestrator = ParallelTaskOrchestrator(
            max_parallel=4,
            role_pool=role_pool,
            retry_config=retry_config,
        )
        assert orchestrator.max_parallel == 4
        assert len(orchestrator.role_pool) == 2
        assert orchestrator.retry_config.max_retries == 3

    @pytest.mark.asyncio
    async def test_run_all_empty_tasks(self):
        """Test run_all with empty task list."""
        orchestrator = ParallelTaskOrchestrator(max_parallel=2)
        results = await orchestrator.run_all([])
        assert results == []

    @pytest.mark.asyncio
    async def test_run_all_single_task_success(self):
        """Test run_all with single successful task."""
        orchestrator = ParallelTaskOrchestrator(max_parallel=1)

        async def success_task():
            return "success"

        tasks = [("task-1", success_task, (), {})]
        results = await orchestrator.run_all(tasks)

        assert len(results) == 1
        assert results[0].task_id == "task-1"
        assert results[0].success is True
        assert results[0].result == "success"
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_run_all_single_task_failure(self):
        """Test run_all with single failing task."""
        orchestrator = ParallelTaskOrchestrator(
            max_parallel=1,
            retry_config=RetryConfig(max_retries=0),  # No retries
        )

        async def failing_task():
            raise ValueError("Task failed")

        tasks = [("task-1", failing_task, (), {})]
        results = await orchestrator.run_all(tasks)

        assert len(results) == 1
        assert results[0].task_id == "task-1"
        assert results[0].success is False
        assert results[0].error == "Task failed"

    @pytest.mark.asyncio
    async def test_run_all_multiple_tasks(self):
        """Test run_all with multiple tasks."""
        orchestrator = ParallelTaskOrchestrator(max_parallel=2)

        async def task_func(value):
            return value * 2

        tasks = [
            ("task-1", task_func, (1,), {}),
            ("task-2", task_func, (2,), {}),
            ("task-3", task_func, (3,), {}),
        ]
        results = await orchestrator.run_all(tasks)

        assert len(results) == 3
        results_by_id = {r.task_id: r for r in results}
        assert results_by_id["task-1"].result == 2
        assert results_by_id["task-2"].result == 4
        assert results_by_id["task-3"].result == 6

    @pytest.mark.asyncio
    async def test_role_assignment(self):
        """Test that roles are assigned to tasks."""
        role_pool = RolePoolManager(roles=["role-a", "role-b"])
        orchestrator = ParallelTaskOrchestrator(
            max_parallel=1,
            role_pool=role_pool,
        )

        assigned_roles = []

        async def capture_role_task(assigned_role=None):
            assigned_roles.append(assigned_role)
            return assigned_role

        tasks = [
            ("task-1", capture_role_task, (), {}),
            ("task-2", capture_role_task, (), {}),
            ("task-3", capture_role_task, (), {}),
        ]
        await orchestrator.run_all(tasks)

        # Roles should be assigned round-robin
        assert assigned_roles == ["role-a", "role-b", "role-a"]

    @pytest.mark.asyncio
    async def test_duration_tracking(self):
        """Test that task duration is tracked."""
        orchestrator = ParallelTaskOrchestrator(max_parallel=1)

        async def slow_task():
            await asyncio.sleep(0.1)
            return "done"

        tasks = [("task-1", slow_task, (), {})]
        results = await orchestrator.run_all(tasks)

        assert results[0].duration >= 0.1


# =============================================================================
# Property-Based Tests for ParallelTaskOrchestrator
# =============================================================================


class TestParallelTaskOrchestratorPropertyBased:
    """
    Property-based tests for ParallelTaskOrchestrator.

    **Feature: parallel-execution**
    **Property 1: Max Parallel Limit Enforcement**
    **Property 7: Error Isolation**
    **Property 8: Failure Reporting Completeness**
    **Validates: Requirements 1.1, 1.2, 6.1, 6.2, 6.3, 6.4**
    """

    @given(
        max_parallel=st.integers(min_value=1, max_value=10),
        num_tasks=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=500)
    def test_property_1_max_parallel_limit_enforcement(self, max_parallel, num_tasks):
        """
        Property 1: Max Parallel Limit Enforcement

        *For any* max_parallel value N and any number of tasks T, the number of
        concurrently executing tasks SHALL never exceed N at any point during execution.

        **Validates: Requirements 1.1, 1.2, 1.4**
        """
        # Track concurrent execution count
        concurrent_count = 0
        max_concurrent_observed = 0
        lock = asyncio.Lock()

        async def tracking_task():
            nonlocal concurrent_count, max_concurrent_observed
            async with lock:
                concurrent_count += 1
                max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            
            # Simulate some work
            await asyncio.sleep(0.01)
            
            async with lock:
                concurrent_count -= 1
            
            return "done"

        orchestrator = ParallelTaskOrchestrator(max_parallel=max_parallel)
        tasks = [(f"task-{i}", tracking_task, (), {}) for i in range(num_tasks)]

        # Run the tasks
        asyncio.run(orchestrator.run_all(tasks))

        # Verify max concurrent never exceeded max_parallel
        assert max_concurrent_observed <= max_parallel, (
            f"Max concurrent {max_concurrent_observed} exceeded max_parallel {max_parallel}"
        )

    @given(
        num_tasks=st.integers(min_value=2, max_value=15),
        fail_indices=st.lists(
            st.integers(min_value=0, max_value=14),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_error_isolation(self, num_tasks, fail_indices):
        """
        Property 7: Error Isolation

        *For any* set of tasks where some tasks fail after all retries, the remaining
        tasks SHALL complete execution and their results SHALL be collected.

        **Validates: Requirements 6.1, 6.2, 6.4**
        """
        # Ensure fail_indices are within bounds
        fail_indices = [i for i in fail_indices if i < num_tasks]
        if not fail_indices:
            fail_indices = [0]  # Ensure at least one failure

        fail_set = set(fail_indices)

        async def task_func(task_index):
            if task_index in fail_set:
                raise ValueError(f"Task {task_index} failed")
            return f"success-{task_index}"

        orchestrator = ParallelTaskOrchestrator(
            max_parallel=3,
            retry_config=RetryConfig(max_retries=0),  # No retries for faster test
        )
        tasks = [
            (f"task-{i}", task_func, (i,), {})
            for i in range(num_tasks)
        ]

        results = asyncio.run(orchestrator.run_all(tasks))

        # All tasks should have results
        assert len(results) == num_tasks

        # Check that successful tasks completed
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        # Number of failures should match fail_indices
        assert len(failed_results) == len(fail_set)

        # Number of successes should be total - failures
        assert len(successful_results) == num_tasks - len(fail_set)

        # Verify successful tasks have correct results
        for result in successful_results:
            task_index = int(result.task_id.split("-")[1])
            assert result.result == f"success-{task_index}"

    @given(
        num_tasks=st.integers(min_value=1, max_value=15),
        num_failures=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_8_failure_reporting_completeness(self, num_tasks, num_failures):
        """
        Property 8: Failure Reporting Completeness

        *For any* run with F failed tasks, the final report SHALL contain exactly F
        failure entries with their corresponding error messages.

        **Validates: Requirements 6.3, 7.3**
        """
        # Ensure num_failures doesn't exceed num_tasks
        actual_failures = min(num_failures, num_tasks)
        fail_set = set(range(actual_failures))

        error_messages = {}
        for i in fail_set:
            error_messages[i] = f"Error message for task {i}"

        async def task_func(task_index):
            if task_index in fail_set:
                raise ValueError(error_messages[task_index])
            return f"success-{task_index}"

        orchestrator = ParallelTaskOrchestrator(
            max_parallel=3,
            retry_config=RetryConfig(max_retries=0),
        )
        tasks = [
            (f"task-{i}", task_func, (i,), {})
            for i in range(num_tasks)
        ]

        results = asyncio.run(orchestrator.run_all(tasks))

        # Count failures
        failed_results = [r for r in results if not r.success]

        # Exactly F failures should be reported
        assert len(failed_results) == actual_failures, (
            f"Expected {actual_failures} failures, got {len(failed_results)}"
        )

        # Each failure should have an error message
        for result in failed_results:
            assert result.error is not None
            assert len(result.error) > 0
            task_index = int(result.task_id.split("-")[1])
            assert error_messages[task_index] in result.error

    @given(
        max_parallel=st.integers(min_value=1, max_value=5),
        num_tasks=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_tasks_complete(self, max_parallel, num_tasks):
        """
        Additional property: All tasks complete

        *For any* set of tasks, all tasks SHALL complete (success or failure)
        and results SHALL be returned for each task.
        """
        async def simple_task(value):
            return value

        orchestrator = ParallelTaskOrchestrator(max_parallel=max_parallel)
        tasks = [
            (f"task-{i}", simple_task, (i,), {})
            for i in range(num_tasks)
        ]

        results = asyncio.run(orchestrator.run_all(tasks))

        # All tasks should have results
        assert len(results) == num_tasks

        # All task IDs should be present
        result_ids = {r.task_id for r in results}
        expected_ids = {f"task-{i}" for i in range(num_tasks)}
        assert result_ids == expected_ids

        # All should be successful
        assert all(r.success for r in results)


# =============================================================================
# Integration Tests for Parallel Execution
# =============================================================================


class TestParallelExecutionIntegration:
    """
    Integration tests for parallel execution with mock LLM responses.

    Tests the full integration between ParallelTaskOrchestrator,
    ConversationRunner, and Evaluator with max_parallel > 1.

    **Validates: Requirements 1.1, 1.2**
    """

    @pytest.mark.asyncio
    async def test_parallel_conversation_generation(self):
        """Test parallel conversation generation with multiple tasks."""
        from unittest.mock import MagicMock, patch, AsyncMock
        from patient_agent_bench.runner.conversation_runner import ConversationResult
        from patient_agent_bench.runner.conversation import Conversation

        # Create mock conversation results
        def create_mock_result(entry_id):
            from langchain_core.messages import AIMessage, HumanMessage
            conv = Conversation([
                HumanMessage(content=f"Hello from {entry_id}"),
                AIMessage(content=f"Response for {entry_id}"),
            ])
            return ConversationResult(
                case_id=entry_id,
                conversation=conv,
                user_profile=f"<profile>{entry_id}</profile>",
                scenario=f"scenario_{entry_id}",
                num_turns=1,
            )

        # Track concurrent execution
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_run_conversation_async(entry, max_turns=None, assigned_role=None):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            # Simulate some work
            await asyncio.sleep(0.05)

            async with lock:
                concurrent_count -= 1

            return create_mock_result(entry.id)

        # Create orchestrator with max_parallel=3
        orchestrator = ParallelTaskOrchestrator(
            max_parallel=3,
            retry_config=RetryConfig(max_retries=0),
        )

        # Create mock entries
        mock_entries = []
        for i in range(6):
            mock_entry = MagicMock()
            mock_entry.id = f"entry_{i}"
            mock_entries.append(mock_entry)

        # Build tasks
        tasks = [
            (entry.id, mock_run_conversation_async, (entry, 3), {})
            for entry in mock_entries
        ]

        # Run tasks
        results = await orchestrator.run_all(tasks)

        # Verify all tasks completed
        assert len(results) == 6
        assert all(r.success for r in results)

        # Verify parallelism was used (max_concurrent should be > 1)
        assert max_concurrent > 1, f"Expected parallel execution, but max_concurrent was {max_concurrent}"
        assert max_concurrent <= 3, f"Max concurrent {max_concurrent} exceeded max_parallel 3"

        # Verify results contain correct data
        for result in results:
            assert result.result is not None
            assert result.result.case_id.startswith("entry_")

    @pytest.mark.asyncio
    async def test_parallel_evaluation(self):
        """Test parallel evaluation with multiple conversations."""
        from unittest.mock import MagicMock

        # Track concurrent execution
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_evaluate_async(conversation, user_profile, scenario, assigned_role=None):
            nonlocal concurrent_count, max_concurrent
            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            # Simulate evaluation work
            await asyncio.sleep(0.05)

            async with lock:
                concurrent_count -= 1

            return {
                "rubric_scores": {"task_completion": 2, "clinical_safety": 2},
                "aggregate_score": 4.0,
                "safety_pass": True,
                "summary": f"Evaluation complete",
            }

        # Create orchestrator with max_parallel=4
        orchestrator = ParallelTaskOrchestrator(
            max_parallel=4,
            retry_config=RetryConfig(max_retries=0),
        )

        # Create mock conversations
        conversations = [
            {
                "case_id": f"conv_{i}",
                "conversation": [{"role": "user", "content": f"Message {i}"}],
                "user_profile": f"<profile>{i}</profile>",
                "scenario": f"scenario_{i}",
            }
            for i in range(8)
        ]

        # Build tasks
        tasks = [
            (
                conv["case_id"],
                mock_evaluate_async,
                (conv["conversation"], conv["user_profile"], conv["scenario"]),
                {},
            )
            for conv in conversations
        ]

        # Run tasks
        results = await orchestrator.run_all(tasks)

        # Verify all tasks completed
        assert len(results) == 8
        assert all(r.success for r in results)

        # Verify parallelism was used
        assert max_concurrent > 1, f"Expected parallel execution, but max_concurrent was {max_concurrent}"
        assert max_concurrent <= 4, f"Max concurrent {max_concurrent} exceeded max_parallel 4"

        # Verify evaluation results
        for result in results:
            assert result.result is not None
            assert result.result["aggregate_score"] == 4.0
            assert result.result["safety_pass"] is True

    @pytest.mark.asyncio
    async def test_parallel_mixed_success_failure(self):
        """Test parallel execution with mixed success and failure results."""
        fail_indices = {1, 3, 5}

        async def mock_task(task_index, assigned_role=None):
            await asyncio.sleep(0.02)
            if task_index in fail_indices:
                raise ValueError(f"Task {task_index} failed intentionally")
            return {"task_index": task_index, "status": "success"}

        orchestrator = ParallelTaskOrchestrator(
            max_parallel=3,
            retry_config=RetryConfig(max_retries=0),
        )

        tasks = [
            (f"task_{i}", mock_task, (i,), {})
            for i in range(7)
        ]

        results = await orchestrator.run_all(tasks)

        # Verify all tasks have results
        assert len(results) == 7

        # Count successes and failures
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        assert len(successes) == 4  # 7 - 3 failures
        assert len(failures) == 3

        # Verify failure error messages
        for result in failures:
            assert "failed intentionally" in result.error

        # Verify success results
        for result in successes:
            assert result.result["status"] == "success"

    @pytest.mark.asyncio
    async def test_parallel_with_role_distribution(self):
        """Test parallel execution with role pool distribution."""
        role_pool = RolePoolManager(roles=["role-1", "role-2", "role-3"])
        assigned_roles_log = []
        lock = asyncio.Lock()

        async def mock_task_with_role(task_id, assigned_role=None):
            async with lock:
                assigned_roles_log.append((task_id, assigned_role))
            await asyncio.sleep(0.02)
            return {"task_id": task_id, "role": assigned_role}

        orchestrator = ParallelTaskOrchestrator(
            max_parallel=2,
            role_pool=role_pool,
            retry_config=RetryConfig(max_retries=0),
        )

        tasks = [
            (f"task_{i}", mock_task_with_role, (f"task_{i}",), {})
            for i in range(6)
        ]

        results = await orchestrator.run_all(tasks)

        # Verify all tasks completed
        assert len(results) == 6
        assert all(r.success for r in results)

        # Verify roles were assigned
        roles_used = [role for _, role in assigned_roles_log]
        assert all(role is not None for role in roles_used)

        # Verify round-robin distribution (each role should be used twice)
        from collections import Counter
        role_counts = Counter(roles_used)
        assert role_counts["role-1"] == 2
        assert role_counts["role-2"] == 2
        assert role_counts["role-3"] == 2

    @pytest.mark.asyncio
    async def test_parallel_respects_max_parallel_under_load(self):
        """Test that max_parallel is strictly enforced under heavy load."""
        max_parallel = 2
        num_tasks = 20
        concurrent_count = 0
        max_concurrent_observed = 0
        violations = []
        lock = asyncio.Lock()

        async def heavy_task(task_id, assigned_role=None):
            nonlocal concurrent_count, max_concurrent_observed
            async with lock:
                concurrent_count += 1
                if concurrent_count > max_parallel:
                    violations.append(f"Violation at task {task_id}: {concurrent_count} concurrent")
                max_concurrent_observed = max(max_concurrent_observed, concurrent_count)

            # Simulate variable work duration
            await asyncio.sleep(0.01 + (hash(task_id) % 5) * 0.01)

            async with lock:
                concurrent_count -= 1

            return {"task_id": task_id}

        orchestrator = ParallelTaskOrchestrator(
            max_parallel=max_parallel,
            retry_config=RetryConfig(max_retries=0),
        )

        tasks = [
            (f"task_{i}", heavy_task, (f"task_{i}",), {})
            for i in range(num_tasks)
        ]

        results = await orchestrator.run_all(tasks)

        # Verify all tasks completed
        assert len(results) == num_tasks
        assert all(r.success for r in results)

        # Verify no violations occurred
        assert len(violations) == 0, f"Concurrency violations: {violations}"

        # Verify max_parallel was respected
        assert max_concurrent_observed <= max_parallel, (
            f"Max concurrent {max_concurrent_observed} exceeded max_parallel {max_parallel}"
        )

    def test_experiment_runner_parallel_mode_initialization(
        self, bench_config, temp_output_dir, temp_benchmark_file
    ):
        """Test ExperimentRunner initializes correctly in parallel mode."""
        from patient_agent_bench.runner.experiment_runner import ExperimentRunner
        from patient_agent_bench.runner.output_manager import OutputManager

        output_mgr = OutputManager(str(temp_benchmark_file), str(temp_output_dir))

        # Test with max_parallel > 1
        runner = ExperimentRunner(
            bench_config,
            output_mgr,
            max_parallel=4,
        )

        assert runner.max_parallel == 4
        assert runner.role_pool is not None
        assert runner.retry_config is not None

    def test_experiment_runner_with_role_pool(
        self, bench_config, temp_output_dir, temp_benchmark_file
    ):
        """Test ExperimentRunner with custom role pool."""
        from patient_agent_bench.runner.experiment_runner import ExperimentRunner
        from patient_agent_bench.runner.output_manager import OutputManager

        output_mgr = OutputManager(str(temp_benchmark_file), str(temp_output_dir))
        role_pool = RolePoolManager(roles=["role-a", "role-b"])

        runner = ExperimentRunner(
            bench_config,
            output_mgr,
            max_parallel=2,
            role_pool=role_pool,
        )

        assert runner.max_parallel == 2
        assert len(runner.role_pool) == 2

    def test_experiment_runner_with_custom_retry_config(
        self, bench_config, temp_output_dir, temp_benchmark_file
    ):
        """Test ExperimentRunner with custom retry configuration."""
        from patient_agent_bench.runner.experiment_runner import ExperimentRunner
        from patient_agent_bench.runner.output_manager import OutputManager

        output_mgr = OutputManager(str(temp_benchmark_file), str(temp_output_dir))
        retry_config = RetryConfig(max_retries=3, base_delay=0.5)

        runner = ExperimentRunner(
            bench_config,
            output_mgr,
            max_parallel=2,
            retry_config=retry_config,
        )

        assert runner.retry_config.max_retries == 3
        assert runner.retry_config.base_delay == 0.5
