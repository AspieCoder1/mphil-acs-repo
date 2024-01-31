from dataclasses import field, dataclass

import hydra
import torch
from hydra.core.config_store import ConfigStore

from core.datasets import get_dataset_lp, LinkPredDatasets
from core.models import get_sheaf_model
from core.sheaf_configs import SheafModelCfg, SheafLinkPredDatasetCfg
from core.trainer import TrainerArgs
from models.SheafGNN.config import SheafModelArguments
from models.SheafLinkPredictor import SheafLinkPredictor
from sheaf_nc import init_trainer


@dataclass
class Config:
    trainer: TrainerArgs = field(default_factory=TrainerArgs)
    tags: list[str] = field(default_factory=list)
    model: SheafModelCfg = field(default_factory=SheafModelCfg)
    dataset: SheafLinkPredDatasetCfg = field(default_factory=SheafLinkPredDatasetCfg)
    model_args: SheafModelArguments = field(default_factory=SheafModelArguments)


cs = ConfigStore.instance()
cs.store("base_config", Config)


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config_lp")
def main(cfg: Config):
    torch.set_float32_matmul_precision("high")
    dm = get_dataset_lp(LinkPredDatasets.LastFM, True)
    dm.prepare_data()

    cfg.model_args.graph_size = dm.graph_size
    cfg.model_args.input_dim = dm.in_channels
    cfg.model_args.output_dim = 64

    model_cls = get_sheaf_model(cfg.model.type)
    model = model_cls(None, cfg.model_args)

    print(model.hidden_dim)

    sheaf_lp = SheafLinkPredictor(
        model=model, num_classes=1,
        hidden_dim=model.hidden_dim
    )

    trainer = init_trainer(cfg)

    trainer.fit(sheaf_lp, dm)
    trainer.test(sheaf_lp, dm)


if __name__ == "__main__":
    main()
