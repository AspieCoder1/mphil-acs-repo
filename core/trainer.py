from dataclasses import dataclass


@dataclass
class TrainerArgs:
    accelerator: str
    devices: int
    num_nodes: int
    patience: int
    strategy: str
