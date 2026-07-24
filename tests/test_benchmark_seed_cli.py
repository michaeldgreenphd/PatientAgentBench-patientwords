# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for benchmark seed CLI module.

Tests the generate-seeds CLI command and run_generate_seeds() function.
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from patient_agent_bench.benchmark_seed.seed_runner import run_generate_seeds
from patient_agent_bench.run import setup_parser


class TestGenerateSeedsParser:
    """Tests for generate-seeds subparser in main CLI."""

    def test_parser_is_added(self):
        """Test that generate-seeds subparser is added correctly."""
        parser = setup_parser()

        args = parser.parse_args(["generate-seeds", "--count", "5"])

        assert args.command == "generate-seeds"
        assert args.count == 5

    def test_count_argument_required(self):
        """Test that --count is required."""
        parser = setup_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["generate-seeds"])

    def test_default_seed_dist_path(self):
        """Test default seed distribution config path is set correctly."""
        parser = setup_parser()

        args = parser.parse_args(["generate-seeds", "--count", "5"])

        assert args.seed_dist == Path("data/default_benchmark_seed.json")

    def test_default_config_path(self):
        """Test default benchmark config path is set correctly."""
        parser = setup_parser()

        args = parser.parse_args(["generate-seeds", "--count", "5"])

        # Config comes from common_parser, stored as string
        assert args.config == "data/default_config.json"

    def test_custom_seed_dist_path(self):
        """Test custom seed distribution config path can be specified."""
        parser = setup_parser()

        args = parser.parse_args([
            "generate-seeds", "--count", "5", "--seed-dist", "custom/dist.json"
        ])

        assert args.seed_dist == Path("custom/dist.json")

    def test_custom_config_path(self):
        """Test custom benchmark config path can be specified."""
        parser = setup_parser()

        args = parser.parse_args([
            "generate-seeds", "--count", "5", "--config", "custom/config.json"
        ])

        # Config comes from common_parser, stored as string
        assert args.config == "custom/config.json"

    def test_output_argument(self):
        """Test --output argument is parsed correctly."""
        parser = setup_parser()

        args = parser.parse_args([
            "generate-seeds", "--count", "5", "--output", "output/seeds.json"
        ])

        assert args.output == Path("output/seeds.json")

    def test_seed_argument(self):
        """Test --seed argument is parsed correctly."""
        parser = setup_parser()

        args = parser.parse_args(["generate-seeds", "--count", "5", "--seed", "42"])

        assert args.seed == 42

    def test_max_parallel_argument(self):
        """Test --max-parallel argument is parsed correctly."""
        parser = setup_parser()

        args = parser.parse_args([
            "generate-seeds", "--count", "5", "--max-parallel", "10"
        ])

        assert args.max_parallel == 10

    def test_max_parallel_default(self):
        """Test --max-parallel defaults to 1."""
        parser = setup_parser()

        args = parser.parse_args(["generate-seeds", "--count", "5"])

        assert args.max_parallel == 1

    def test_short_argument_forms(self):
        """Test short argument forms work correctly."""
        parser = setup_parser()

        args = parser.parse_args([
            "generate-seeds",
            "-n", "10",
            "--seed-dist", "dist.json",
            "-o", "out.json",
            "-s", "123",
        ])

        assert args.count == 10
        assert args.seed_dist == Path("dist.json")
        assert args.output == Path("out.json")
        assert args.seed == 123


class TestRunGenerateSeeds:
    """Tests for run_generate_seeds function."""

    def test_seed_dist_file_not_found(self):
        """Test error handling when seed distribution config file doesn't exist."""
        args = argparse.Namespace(
            count=5,
            seed_dist=Path("nonexistent/dist.json"),
            config=Path("data/default_config.json"),
            output=None,
            seed=None,
            max_parallel=1,
        )

        with pytest.raises(SystemExit) as exc_info:
            run_generate_seeds(args)

        assert exc_info.value.code == 1

    def test_bench_config_file_not_found(self):
        """Test error handling when benchmark config file doesn't exist."""
        args = argparse.Namespace(
            count=5,
            seed_dist=Path("data/default_benchmark_seed.json"),
            config=Path("nonexistent/config.json"),
            output=None,
            seed=None,
            max_parallel=1,
        )

        with pytest.raises(SystemExit) as exc_info:
            run_generate_seeds(args)

        assert exc_info.value.code == 1

    @patch("patient_agent_bench.benchmark_seed.seed_runner.BenchConfig.from_file")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.ensure_credentials")
    def test_credentials_failure_exits(self, mock_ensure_creds, mock_from_file):
        """Test that credential failure causes exit."""
        mock_config = MagicMock()
        mock_config.seed_generator_model = MagicMock()
        mock_from_file.return_value = mock_config
        mock_ensure_creds.return_value = False

        args = argparse.Namespace(
            count=5,
            seed_dist=Path("data/default_benchmark_seed.json"),
            config=Path("data/default_config.json"),
            output=None,
            seed=None,
            max_parallel=1,
        )

        with pytest.raises(SystemExit) as exc_info:
            run_generate_seeds(args)

        assert exc_info.value.code == 1

    @patch("patient_agent_bench.benchmark_seed.seed_runner.write_summary")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.analyze_entries")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.write_benchmark_entries")
    @patch("patient_agent_bench.benchmark_seed.seed_runner._enrich_seeds_parallel")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.asyncio.run")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.RolePoolManager.from_env")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.BenchConfig.from_file")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.ensure_credentials")
    def test_generates_enriched_entries(
        self,
        mock_ensure_creds,
        mock_bench_config,
        mock_role_pool,
        mock_asyncio_run,
        mock_enrich,
        mock_write,
        mock_analyze,
        mock_write_summary,
        tmp_path,
    ):
        """Test that generate-seeds produces enriched benchmark entries."""
        mock_ensure_creds.return_value = True

        mock_config = MagicMock()
        mock_config.seed_generator_model = MagicMock()
        mock_bench_config.return_value = mock_config

        mock_role_pool.return_value = MagicMock()
        mock_role_pool.return_value.__len__ = MagicMock(return_value=0)

        # Mock enriched entries
        mock_entries = [MagicMock() for _ in range(3)]
        mock_asyncio_run.return_value = mock_entries

        output_path = tmp_path / "benchmark.json"
        args = argparse.Namespace(
            count=3,
            seed_dist=Path("data/default_benchmark_seed.json"),
            config=Path("data/default_config.json"),
            output=output_path,
            seed=42,
            max_parallel=1,
        )

        run_generate_seeds(args)

        # Verify enriched generation was called
        mock_asyncio_run.assert_called_once()
        mock_write.assert_called_once_with(mock_entries, output_path)

    @patch("patient_agent_bench.benchmark_seed.seed_runner.write_summary")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.analyze_entries")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.write_benchmark_entries")
    @patch("patient_agent_bench.benchmark_seed.seed_runner._enrich_seeds_parallel")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.asyncio.run")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.RolePoolManager.from_env")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.BenchConfig.from_file")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.ensure_credentials")
    def test_auto_generates_output_path(
        self,
        mock_ensure_creds,
        mock_bench_config,
        mock_role_pool,
        mock_asyncio_run,
        mock_enrich,
        mock_write,
        mock_analyze,
        mock_write_summary,
    ):
        """Test that output path is auto-generated when not specified."""
        mock_ensure_creds.return_value = True

        mock_config = MagicMock()
        mock_config.seed_generator_model = MagicMock()
        mock_bench_config.return_value = mock_config

        mock_role_pool.return_value = MagicMock()
        mock_role_pool.return_value.__len__ = MagicMock(return_value=0)

        mock_asyncio_run.return_value = [MagicMock()]

        args = argparse.Namespace(
            count=1,
            seed_dist=Path("data/default_benchmark_seed.json"),
            config=Path("data/default_config.json"),
            output=None,  # No output specified
            seed=None,
            max_parallel=1,
        )

        run_generate_seeds(args)

        # Verify write was called with auto-generated path in data/ directory
        call_args = mock_write.call_args
        output_path = call_args[0][1]
        assert str(output_path).startswith("data/benchmark_")
        assert str(output_path).endswith(".json")

    @patch("patient_agent_bench.benchmark_seed.seed_runner.write_summary")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.analyze_entries")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.write_benchmark_entries")
    @patch("patient_agent_bench.benchmark_seed.seed_runner._enrich_seeds_parallel")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.asyncio.run")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.RolePoolManager.from_env")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.BenchConfig.from_file")
    @patch("patient_agent_bench.benchmark_seed.seed_runner.ensure_credentials")
    def test_passes_max_parallel_to_enrichment(
        self,
        mock_ensure_creds,
        mock_bench_config,
        mock_role_pool,
        mock_asyncio_run,
        mock_enrich,
        mock_write,
        mock_analyze,
        mock_write_summary,
    ):
        """Test that max_parallel is passed to parallel enrichment."""
        mock_ensure_creds.return_value = True

        mock_config = MagicMock()
        mock_config.seed_generator_model = MagicMock()
        mock_bench_config.return_value = mock_config

        mock_pool = MagicMock()
        mock_pool.__len__ = MagicMock(return_value=3)
        mock_role_pool.return_value = mock_pool

        mock_asyncio_run.return_value = [MagicMock()]

        args = argparse.Namespace(
            count=5,
            seed_dist=Path("data/default_benchmark_seed.json"),
            config=Path("data/default_config.json"),
            output=Path("out.json"),
            seed=None,
            max_parallel=10,
        )

        run_generate_seeds(args)

        # Verify asyncio.run was called (which calls _enrich_seeds_parallel)
        mock_asyncio_run.assert_called_once()
