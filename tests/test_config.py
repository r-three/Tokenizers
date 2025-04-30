import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import sleep
from unittest.mock import patch

import pytest
import yaml
from transformers import HfArgumentParser
from xarch_tokenizers.config import Config
from xarch_tokenizers.experiment_config import EvaluationConfig, load_config

from tests.test_experiment import TestExperiment


class TestConfig(TestExperiment):
    SAVE_DIR = Path(__file__).parent.joinpath("test_results")

    @pytest.fixture(scope="function")
    def setup(self):
        """Create a temporary experiment dir for each config test"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"test-{timestamp}"
        with tempfile.TemporaryDirectory(
            dir=self.SAVE_DIR, prefix=experiment_name
        ) as temp_dir:
            yield {"experiment_dir": Path(temp_dir), "experiment_name": experiment_name}

    def test_config_custom_values(self, setup):
        """Test Config initialization with custom values."""
        custom_config = Config(
            model_name="custom/model",
            datasets=["juletxara/mgsm:subset=en", "EleutherAI/hendrycks_math"],
            use_wandb=True,
            wandb_project="custom_project",
            seed=1234,
            batch_size=64,
            tensor_parallel_size=4,
            tokenizer_name="custom/tokenizer",
            save_dir=self.SAVE_DIR,
            experiment_name=setup["experiment_name"],
        )
        # Check custom values
        assert custom_config.model_name == "custom/model"
        assert custom_config.datasets == ["juletxara/mgsm", "EleutherAI/hendrycks_math"]
        assert custom_config.use_wandb is True
        assert custom_config.wandb_project == "custom_project"
        assert custom_config.seed == 1234
        assert custom_config.batch_size == 64
        assert custom_config.tensor_parallel_size == 4
        assert custom_config.tokenizer_name == "custom/tokenizer"

        # Check parsed dataset configs
        dataset_configs = custom_config.get_dataset_configs()
        assert len(dataset_configs) == 2
        assert dataset_configs[0].name == "juletxara/mgsm"
        assert dataset_configs[0].subset == "en"
        assert dataset_configs[1].name == "EleutherAI/hendrycks_math"

    def _test_from_file(self, suffix, setup):
        config_data = {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "datasets": ["juletxara/mgsm:subset=en", "EleutherAI/hendrycks_math"],
            "chat_template": "qwen",
            "batch_size": 16,
            "dtype": "fp16",
            "save_dir": self.SAVE_DIR.as_posix(),
            "experiment_name": setup["experiment_name"],
        }
        if suffix == ".json":
            txt = json.dumps(config_data).encode("utf-8")
        else:
            txt = yaml.dump(config_data, encoding="utf-8")
        with tempfile.NamedTemporaryFile(
            dir=self.SAVE_DIR, suffix=suffix, delete=False
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(txt)

        try:
            # Load config from the file
            config = Config.from_file(temp_path)

            # Check loaded values
            assert config.model_name == "Qwen/Qwen2.5-7B-Instruct"
            assert config.chat_template == "qwen"
            assert config.datasets == ["juletxara/mgsm", "EleutherAI/hendrycks_math"]
            assert config.batch_size == 16
            assert config.dtype == "fp16"
        finally:
            os.unlink(temp_path)

    def test_from_file_json(self, setup):
        """Test loading Config from a JSON file."""
        self._test_from_file(suffix=".json", setup=setup)

    def test_from_file_yaml(self, setup):
        """Test loading Config from a YAML file."""
        self._test_from_file(suffix=".yaml", setup=setup)
        self._test_from_file(suffix=".yml", setup=setup)

    def test_non_existing_file(self):
        """Test handling of non-existing configuration file."""
        non_existent_file = "non_existent_file.yaml"
        with pytest.raises(FileNotFoundError) as excinfo:
            Config.from_file(non_existent_file)

        expected_error = f"{non_existent_file} not found. Please pass an existing file."
        assert str(excinfo.value) == expected_error

    def test_unsupported_file_format(self):
        """Test handling of unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with pytest.raises(ValueError) as excinfo:
                Config.from_file(temp_path)

            expected_error = (
                "File format .txt not supported. Please pass a json, yaml or yml file."
            )
            assert str(excinfo.value) == expected_error
        finally:
            os.unlink(temp_path)

    def test_save_config(self, setup):
        """Test saving Config to files."""
        config = Config(
            model_name="test/model",
            datasets=["juletxara/mgsm:subset=en", "EleutherAI/hendrycks_math"],
            save_dir=self.SAVE_DIR,
            experiment_name=setup["experiment_name"],
        )

        # Save config
        yaml_path = config.save_config()

        # Check that files were created
        assert yaml_path.exists()
        # Check YAML content
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        assert yaml_data["model_name"] == "test/model"
        assert yaml_data["experiment_name"] == config.experiment_name
        assert "dataset_configs" in yaml_data

    def _test_tasks(self, config):
        tasks = config.get_dataset_tasks()

        assert len(tasks) == 3
        assert tasks[0] == "juletxara/mgsm_en"
        assert tasks[1] == "EleutherAI/hendrycks_math"
        assert tasks[2] == "juletxara/mgsm_de"

        datasets = config.datasets
        assert datasets[0] == "juletxara/mgsm"
        assert datasets[1] == "EleutherAI/hendrycks_math"
        assert datasets[2] == "juletxara/mgsm"  # Split doesn't affect dataset name

    def test_get_dataset_tasks(self, setup):
        """Test getting dataset tasks in lm-eval format."""
        config = Config(
            datasets=[
                "juletxara/mgsm:subset=en",
                "EleutherAI/hendrycks_math",
                "juletxara/mgsm:subset=de",
            ],
            save_dir=self.SAVE_DIR,
            experiment_name=setup["experiment_name"],
        )
        self._test_tasks(config)

    def test_config_from_command(self, monkeypatch, setup):
        """test that config options passed through
        the command line are parsed correctly"""
        monkeypatch.setattr("sys.argv", self._get_test_args(setup))
        parser = HfArgumentParser((EvaluationConfig,))
        (config,) = parser.parse_args_into_dataclasses()

        self._test_tasks(config)

    def test_load_config(self, monkeypatch, setup):
        """test the utility function works properly"""
        monkeypatch.setattr("sys.argv", self._get_test_args(setup))
        config = load_config(EvaluationConfig)
        self._test_tasks(config)
        assert config.num_samples == 5
        assert config.model_name == "meta-llama/Llama-3.2-1B-Instruct"
        assert config.use_wandb
