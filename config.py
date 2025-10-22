from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class OptimConfig:
    name: str = 'adam'
    lr: float = 2e-4 
    lr_decay_steps: int = 15
    gamma: float = 0.5

@dataclass
class ModelConfig:
    model: str = 'epit'
    dim: int = 64

@dataclass
class EvaluationConfig:
    batch_size: int = 16
    eval_step: int = 1
    save_lf_step: int = 5

@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 16
    device: str = 'cuda:2'
    name: str = ''
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool)):
        return str(obj)
    else:
        return obj