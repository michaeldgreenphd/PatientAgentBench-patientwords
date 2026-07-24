# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Parallel Task Orchestrator for PatientAgentBench.

Provides concurrent execution of conversations and evaluations with:
- Semaphore-based concurrency control
- Role pool integration for load distribution
- Retry handling with exponential backoff
- Error isolation and comprehensive failure reporting
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from patient_agent_bench.config import RolePoolManager
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.utils.retry import RetryConfig, retry_with_backoff

logger = get_logger(__name__)


@dataclass
class TaskResult:
    """
    Result of a parallel task execution.

    Captures the outcome of a single task including success/failure status,
    the result or error message, and execution duration.

    Attributes:
        task_id: Unique identifier for the task
        success: Whether the task completed successfully
        result: The task result if successful, None otherwise
        error: Error message if task failed, None otherwise
        duration: Execution time in seconds
    """

    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0


class ParallelTaskOrchestrator:
    """
    Orchestrates parallel task execution with concurrency control.

    Uses asyncio.Semaphore to limit concurrent tasks to max_parallel.
    Integrates with RolePoolManager for role distribution and RetryConfig
    for retry handling with exponential backoff.

    Attributes:
        max_parallel: Maximum number of concurrent tasks
        role_pool: Pool of AWS ARN roles for load distribution
        retry_config: Configuration for retry behavior
    """

    def __init__(
        self,
        max_parallel: int = 1,
        role_pool: Optional[RolePoolManager] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the ParallelTaskOrchestrator.

        Args:
            max_parallel: Maximum number of concurrent tasks (default: 1)
            role_pool: Pool of AWS ARN roles for load distribution
            retry_config: Configuration for retry behavior
        """
        self.max_parallel = max_parallel
        self.role_pool = role_pool or RolePoolManager(roles=[])
        self.retry_config = retry_config or RetryConfig()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._completed = 0
        self._failed = 0
        self._lock = asyncio.Lock()


    async def _run_task(
        self,
        task_id: str,
        task_func: Callable[..., Awaitable[Any]],
        total_tasks: int,
        on_complete: Optional[Callable[["TaskResult"], Awaitable[None]]],
        *args,
        **kwargs,
    ) -> TaskResult:
        """
        Run a single task with semaphore control and retry handling.

        Acquires the semaphore before execution, assigns a role from the pool
        if available, executes the task with retry handling, and releases
        the semaphore when done. If on_complete is provided, it is called
        with the TaskResult immediately after the task finishes (before
        gather returns), enabling truly incremental saves.

        Args:
            task_id: Unique identifier for the task
            task_func: Async function to execute
            total_tasks: Total number of tasks (for progress logging)
            on_complete: Optional async callback invoked with the TaskResult
                immediately when this task finishes
            *args: Positional arguments to pass to task_func
            **kwargs: Keyword arguments to pass to task_func

        Returns:
            TaskResult with success/failure status and result/error
        """
        # Semaphore is guaranteed to be set by run_all() before this is called
        assert self._semaphore is not None, "_run_task called without semaphore"

        async with self._semaphore:
            start_time = time.time()

            # Assign role for this task
            role = self.role_pool.get_role()
            if role:
                kwargs["assigned_role"] = role

            try:
                result = await retry_with_backoff(
                    task_func,
                    *args,
                    config=self.retry_config,
                    **kwargs,
                )

                async with self._lock:
                    self._completed += 1
                    logger.info(
                        "Task %s completed (%d/%d)",
                        task_id,
                        self._completed + self._failed,
                        total_tasks,
                    )

                task_result = TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    duration=time.time() - start_time,
                )

            except Exception as e:
                async with self._lock:
                    self._failed += 1
                    logger.error(
                        "Task %s failed: %s (%d/%d)",
                        task_id,
                        str(e),
                        self._completed + self._failed,
                        total_tasks,
                    )

                task_result = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                    duration=time.time() - start_time,
                )

        # Fire callback after semaphore release so it doesn't block concurrency
        if on_complete is not None:
            await on_complete(task_result)

        return task_result


    async def run_all(
        self,
        tasks: List[Tuple[str, Callable[..., Awaitable[Any]], tuple, dict]],
        on_complete: Optional[Callable[["TaskResult"], Awaitable[None]]] = None,
    ) -> List[TaskResult]:
        """
        Run all tasks with parallel execution.

        Creates coroutines for all tasks and uses asyncio.gather to run them
        concurrently. The semaphore limits actual parallelism to max_parallel.
        All results (success and failure) are collected and returned.

        Args:
            tasks: List of (task_id, func, args, kwargs) tuples where:
                - task_id: Unique identifier for the task
                - func: Async function to execute
                - args: Tuple of positional arguments
                - kwargs: Dict of keyword arguments
            on_complete: Optional async callback invoked with each TaskResult
                immediately when that task finishes (before gather returns).
                Enables truly incremental saves.

        Returns:
            List of TaskResult objects for all tasks
        """
        self._semaphore = asyncio.Semaphore(self.max_parallel)
        self._completed = 0
        self._failed = 0

        total_tasks = len(tasks)
        logger.info(
            "Starting %d tasks with max_parallel=%d",
            total_tasks,
            self.max_parallel,
        )

        # Create coroutines for all tasks
        coroutines = [
            self._run_task(task_id, func, total_tasks, on_complete, *args, **kwargs)
            for task_id, func, args, kwargs in tasks
        ]

        # Run all tasks concurrently (semaphore limits actual parallelism)
        results = await asyncio.gather(*coroutines, return_exceptions=False)

        logger.info(
            "Completed %d tasks: %d successful, %d failed",
            total_tasks,
            self._completed,
            self._failed,
        )

        return results
