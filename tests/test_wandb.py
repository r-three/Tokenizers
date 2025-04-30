from unittest.mock import MagicMock, patch

from xarch_tokenizers.logging.wandb_utils import setup_wandb

from .test_experiment import TestExperiment


class TestWandbSetup(TestExperiment):
    """Tests for Weights & Biases setup."""

    def test_setup_wandb_when_enabled(self, setup, monkeypatch):
        """Test wandb initialization when enabled in config."""
        # Get config from setup
        config = setup["config"]
        config.use_wandb = True

        # Mock wandb.init
        with patch("wandb.init") as mock_wandb_init:
            setup_wandb(config)

            # Verify wandb was initialized with correct parameters
            mock_wandb_init.assert_called_once()
            args, kwargs = mock_wandb_init.call_args
            assert kwargs["project"] == config.wandb_project
            assert kwargs["name"] == config.experiment_name
            assert kwargs["entity"] == config.wandb_entity
            assert kwargs["tags"] == config.tags

    def test_setup_wandb_when_disabled(self, setup):
        """Test wandb is not initialized when disabled in config."""
        # Get config from setup
        config = setup["config"]
        config.use_wandb = False

        # Mock wandb.init
        with patch("wandb.init") as mock_wandb_init:
            setup_wandb(config)

            # Verify wandb was not initialized
            mock_wandb_init.assert_not_called()
