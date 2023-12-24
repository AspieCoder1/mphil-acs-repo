from dataclasses import dataclass
from typing import Literal

from node_classification import Datasets


@dataclass
class SheafModelArguments:
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
    add_hb: bool = True
    use_act: bool = True
    second_linear: bool = False
    orth: Literal['matrix_exp', 'cayley', 'householder', 'euler'] = 'householder'
    sheaf_act: str = "tanh"
    edge_weights: bool = True
    sparse_learner: bool = False


@dataclass
class ODEArguments:
    int_method: Literal['dopri5', 'euler', 'rk4', 'midpoint']
    max_t: float = 1.0
    step_size: float = 1.0
    max_iters: int = 100
    adjoint_method: str = 'adaptive_heun'
    adjoint: bool = False
    adjoint_step_size: float = 1.0
    tol_scale: float = 1.0
    tol_scale_adjoint: float = 1.0
    max_nfe: int = 1000
    no_early: bool = False
    earlystopxT: float = 3.0
    max_test_steps: int = 100


@dataclass
class Config:
    model_args: SheafModelArguments
    ode_args: ODEArguments
    model: Literal[
        'DiagSheaf', 'BundleSheaf', 'GeneralSheaf', 'DiagSheafODE', 'BundleSheafODE', 'GeneralSheafODE']
    dataset: Datasets
