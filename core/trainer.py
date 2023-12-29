from dataclasses import dataclass


@dataclass
class TrainerArgs:
    accelerator: str = "gpu"
    devices: int = 1
    num_nodes: int = 1
    patience: int = 100
    strategy: str = "auto"
    fast_dev_run: bool = False
