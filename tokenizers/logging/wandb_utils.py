from dataclasses import asdict

import wandb


def setup_wandb(config):
    """Initialize wandb if enabled in config."""
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.experiment_name,
            entity=config.wandb_entity,
            tags=config.tags,
            config=asdict(config),
        )
