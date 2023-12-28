from dataclasses import dataclass


@dataclass
class Trainer:
    accelerator: str
    devices: int
    num_nodes: int
    patience: int
    strategy: str
