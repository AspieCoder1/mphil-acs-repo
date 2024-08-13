#  Copyright (c) 2024. Luke Braithwaite
#  Adapted from: https://github.com/twitter-research/neural-sheaf-diffusion

from dataclasses import dataclass, field
from enum import auto
from typing import Optional

from strenum import LowercaseStrEnum, SnakeCaseStrEnum


class IntMethod(LowercaseStrEnum):
    dopri5 = auto()
    euler = auto()
    rk4 = auto()
    midpoint = auto()


class OrthogonalMethod(LowercaseStrEnum):
    matrix_exp = auto()
    cayley = auto()
    householder = auto()
    euler = auto()


@dataclass
class ODEArguments:
    int_method: IntMethod = IntMethod.dopri5
    max_t: float = 1.0
    step_size: float = 1.0
    max_iters: int = 100
    adjoint_method: str = "adaptive_heun"
    adjoint: bool = False
    adjoint_step_size: float = 1.0
    tol_scale: float = 1.0
    tol_scale_adjoint: float = 1.0
    max_nfe: int = 1000
    no_early: bool = False
    earlystopxT: float = 3.0
    max_test_steps: int = 100


@dataclass
class SheafModelArguments:
    # ode args
    ode_args: ODEArguments = field(default_factory=ODEArguments)
    d: int = 2
    layers: int = 2
    normalised: bool = True
    deg_normalised: bool = False
    linear: bool = True
    hidden_channels: int = 20
    input_dropout: float = 0.0
    left_weights: bool = True
    right_weights: bool = True
    add_lp: bool = True
    add_hp: bool = True
    use_act: bool = True
    second_linear: bool = False
    orth: OrthogonalMethod = OrthogonalMethod.householder
    sheaf_act: str = "tanh"
    edge_weights: bool = True
    sparse_learner: bool = False
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    device: str = "cuda"
    graph_size: Optional[int] = None
    dropout: float = 0.0
    initial_dropout: float = 0.0
    num_edge_types: Optional[int] = None
    num_node_types: Optional[int] = None
    use_hidden_embeddings: bool = False


@dataclass
class IndSheafModelArguments(SheafModelArguments):
    edge_weights: bool = False


class SheafLearners(SnakeCaseStrEnum):
    type_concat = auto()
    local_concat = auto()
    type_ensemble = auto()
    edge_type_concat = auto()
    node_type_concat = auto()
    node_type = auto()
    edge_type = auto()
    node_edge_types = auto()
