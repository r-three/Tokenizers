## Base class for all tests
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from tokenizers.experiment_config import EvaluationConfig, load_config


class TestExperiment:
    COMMAND_STR = " --datasets juletxara/mgsm:subset=en EleutherAI/hendrycks_math \
            juletxara/mgsm:subset=de --num_samples 5 \
            --model_name meta-llama/Llama-3.2-1B-Instruct \
            --use_wandb=True \
            --save_dir=./test_results "
    PYTHON_COMMAND = "uv run eval"
    SAVE_DIR = Path(__file__).parent.joinpath("test_results")
    config_class = EvaluationConfig

    def _get_test_args(self, setup):
        COMMAND_STR = f""" 
        --datasets juletxara/mgsm:subset=en \
            EleutherAI/hendrycks_math \
            juletxara/mgsm:subset=de \
        --num_samples 5 \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --use_wandb=True \
        --save_dir={self.SAVE_DIR} \
        --experiment_name={setup["experiment_name"]}
        """
        testargs = ["uv run eval"] + COMMAND_STR.split()
        return testargs

    @pytest.fixture(scope="session", autouse=True)
    def cleanup_save_dir(self):
        """Create SAVE_DIR before tests and delete it afterward."""
        # Setup: ensure the directory exists
        self.SAVE_DIR.mkdir(parents=True, exist_ok=True)

        # Let tests run
        yield

        # Teardown: remove the directory after all tests
        if self.SAVE_DIR.exists():
            print(f"Cleaning up test directory: {self.SAVE_DIR}")
            try:
                import shutil

                shutil.rmtree(self.SAVE_DIR)
            except Exception as e:
                print(f"Failed to clean up directory {self.SAVE_DIR}: {e}")

    @pytest.fixture(scope="function")
    def setup(self, monkeypatch):
        """Create a temporary experiment dir for each config test"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"test-{timestamp}"
        with tempfile.TemporaryDirectory(
            dir=self.SAVE_DIR, prefix=experiment_name
        ) as temp_dir:
            testargs = self._get_test_args({"experiment_name": experiment_name})
            monkeypatch.setattr("sys.argv", testargs)
            config = load_config(self.config_class)
            yield {
                "experiment_dir": Path(temp_dir),
                "experiment_name": experiment_name,
                "config": config,
            }
