"""Training configuration and hyperparameter defaults for Balatro PPO."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class TrainingConfig:
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Network architecture
    features_dim: int = 512
    net_arch_pi: list = field(default_factory=lambda: [256, 256])
    net_arch_vf: list = field(default_factory=lambda: [256, 256])

    # Training
    total_timesteps: int = 10_000_000
    n_envs: int = 4
    seed: int = 42

    # Curriculum
    use_curriculum: bool = True
    curriculum_max_ante: int = 8

    # Logging
    log_dir: str = "logs"
    model_dir: str = "models"
    tensorboard_log: str = "logs/tensorboard"
    checkpoint_freq: int = 50_000
    eval_freq: int = 25_000
    log_freq: int = 1_000
    save_freq: int = 100_000

    # Misc
    device: str = "cpu"
    use_vec_normalize: bool = False
    progress_bar: bool = False
    verbose: int = 1

    def to_sb3_kwargs(self) -> Dict[str, Any]:
        """Convert to Stable-Baselines3 PPO constructor kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "policy_kwargs": {
                "features_extractor_class": None,  # set by caller
                "features_extractor_kwargs": {"features_dim": self.features_dim},
                "net_arch": {
                    "pi": self.net_arch_pi,
                    "vf": self.net_arch_vf,
                },
            },
            "device": self.device,
            "verbose": self.verbose,
        }


# Default config for quick test
QUICK_TEST_CONFIG = TrainingConfig(
    total_timesteps=10_000,
    n_envs=2,
    n_steps=512,
    batch_size=32,
    n_epochs=4,
    checkpoint_freq=1_000,
    eval_freq=1_000,
    log_freq=200,
    save_freq=2_000,
    use_vec_normalize=False,
    features_dim=128,
    net_arch_pi=[128, 128],
    net_arch_vf=[128, 128],
    progress_bar=False,
)

# Default config for server training (GPU + many envs)
SERVER_TRAIN_CONFIG = TrainingConfig(
    total_timesteps=5_000_000,
    n_envs=8,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    features_dim=512,
    net_arch_pi=[256, 256],
    net_arch_vf=[256, 256],
    use_curriculum=True,
    checkpoint_freq=100_000,
    eval_freq=50_000,
    log_freq=5_000,
    save_freq=500_000,
    device="cuda",
)
