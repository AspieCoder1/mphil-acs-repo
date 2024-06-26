#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerArgs:
    accelerator: str = "gpu"
    devices: int = 1
    num_nodes: int = 1
    patience: int = 100
    strategy: str = "auto"
    fast_dev_run: bool = False
    log_every_n_steps: int = 1
    max_epochs: int = 100
    logger: bool = True
    profiler: Optional[str] = None
