import json
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from .test_experiment import TestExperiment


class TestEnvironmentSetup(TestExperiment):
    """Tests for environment setup."""

    def test_setup_environment(self, setup, monkeypatch):
        """Test environment setup including model and tokenizer loading."""
        # Get config from setup
        config = setup["config"]

        # Mock dependencies
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_logger = MagicMock()

        with patch(
            "ca.logging.logger.setup_logger", return_value=mock_logger
        ) as mock_setup_logger, patch(
            "ca.models.load_tokenizer", return_value=mock_tokenizer
        ) as mock_load_tokenizer, patch(
            "lm_eval.models.huggingface.HFLM", return_value=mock_model
        ) as mock_hflm, patch("torch.cuda.is_available", return_value=True):
            # Call the function to test
            from xarch_tokenizers.scripts.eval import setup_lm_eval_environment

            model, tokenizer, logger, device = setup_lm_eval_environment(config)

            # Verify logger setup
            mock_setup_logger.assert_called_once_with(config, "eval")

            # Verify tokenizer and model loading
            mock_load_tokenizer.assert_called_once()
            mock_hflm.assert_called_once()

            # Verify correct return values
            assert model == mock_model
            assert tokenizer == mock_tokenizer
            assert logger == mock_logger
            assert device == "cuda"


class TestEvaluation(TestExperiment):
    """Tests for evaluation functionality with real lm_eval calls."""

    COMMAND_STR = " --datasets openai/gsm8k --num_samples 2 \
            --load_existing=True \
            --model_name meta-llama/Llama-3.2-1B-Instruct \
            --use_wandb=True \
            --save_dir=./test_results "

    # @pytest.mark.skipif(
    #     not torch.cuda.is_available(),
    #     reason="CUDA not available, skipping integration test",
    # )
    @pytest.mark.gpu
    def test_run_evaluation_integration(self, setup):
        """
        Integration test running actual evaluation with minimal data.
        This test calls the real lm_eval functions with a simple model.
        """
        # Get config from setup
        config = setup["config"]
        from xarch_tokenizers.scripts.eval import (
            run_evaluation,
            setup_lm_eval_environment,
        )

        model, tokenizer, logger, device = setup_lm_eval_environment(config)
        results, tasks = run_evaluation(config, model)

        # Verify we got results in the expected format
        assert "results" in results
        assert len(tasks) > 0
        assert len(results["results"]) > 0


class TestResultsSaving(TestExperiment):
    """Tests for saving evaluation results."""

    def test_save_results(self, setup):
        """Test saving results to CSV and JSON files."""
        # Get config from setup
        config = setup["config"]

        # Set up test data
        processed_results = [
            {"dataset": "dataset1", "model": "model1", "metric1": 0.9, "metric2": 0.8},
            {"dataset": "dataset2", "model": "model1", "metric1": 0.7, "metric2": 0.6},
        ]

        # Call the function to test
        from xarch_tokenizers.scripts.eval import save_results

        save_results(processed_results, config)

        # Verify files were created
        csv_path = config.experiment_dir / "results.csv"
        json_path = config.experiment_dir / "results.json"
        assert csv_path.exists()
        assert json_path.exists()

        # Verify CSV content
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "dataset" in df.columns
        assert "metric1" in df.columns

        # Verify JSON content
        with open(json_path, "r") as f:
            json_data = json.load(f)
        assert "config" in json_data
        assert "results" in json_data
        assert len(json_data["results"]) == 2
