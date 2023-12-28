from dataclasses import dataclass
from enum import auto
from typing import Literal, Optional

import hydra
from hydra.core.config_store import ConfigStore
from strenum import PascalCaseStrEnum

from core.datasets import NCDatasets, get_dataset_nc
from core.trainer import Trainer


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
    sheaf_act: str = 'tanh'
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


class Model(PascalCaseStrEnum):
    DiagSheaf = auto()
    BundleSheaf = auto()
    GeneralSheaf = auto()
    DiagSheafODE = auto()
    BundleSheafODE = auto()
    GeneralSheafODE = auto()


@dataclass
class Config:
    model_args: Optional[SheafModelArguments]
    ode_args: Optional[ODEArguments]
    model: Model
    dataset: NCDatasets
    trainer: Trainer


cs = ConfigStore.instance()
cs.store("config", Config)


@hydra.main(version_base=None, config_path="configs", config_name="sheaf_config.yaml")
def main(cfg: Config) -> None:
    ...
    # 1) get the datamodule
    # The data  must be homogeneous due to how code is configured
    datamodule = get_dataset_nc(cfg.dataset, True)
    datamodule.prepare_data()
    # 2) initialise model
    print(datamodule.edge_index)
    # model = DiscreteBundleSheafDiffusion(cfg.model_args, datamodule.edge_index)
    # 3) initialise trainer
    # 4) train the model
    # 5) test the model


if __name__ == '__main__':
    main()
