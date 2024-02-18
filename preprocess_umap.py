import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from torch_geometric.data import Data

from core.datasets import get_dataset_nc
from core.models import get_sheaf_model
from core.sheaf_configs import ModelTypes
from models.sheaf_gnn.orthogonal import Orthogonal
from models.sheaf_gnn.sheaf_base import SheafDiffusion
from models.sheaf_node_classifier import SheafNodeClassifier
from sheaf_nc import Config

cs = ConfigStore.instance()
cs.store("base_config", Config)

checkpoint_paths = {
    "DiagSheaf-DBLP": "sheafnc_checkpoints/kj4z929k/DiagSheaf-DBLP-epoch=191.ckpt",
    "DiagSheaf-ACM": "sheafnc_checkpoints/6qed03f2/DiagSheaf-ACM-epoch=139.ckpt",
    "DiagSheaf-IMDB": "sheafnc_checkpoints/sxg6vny7/DiagSheaf-IMDB-epoch=50.ckpt",
    "BundleSheaf-IMDB": "sheafnc_checkpoints/6q2u4tre/BundleSheaf-IMDB-epoch=37.ckpt",
    "BundleSheaf-ACM": "sheafnc_checkpoints/650bmz3e/BundleSheaf-ACM-epoch=173.ckpt",
    "BundleSheaf-DBLP": "sheafnc_checkpoints/e40uef9h/BundleSheaf-DBLP-epoch=96.ckpt",
    "GeneralSheaf-IMDB": "sheafnc_checkpoints/xep1bpuf/GeneralSheaf-IMDB-epoch=52.ckpt",
    "GeneralSheaf-ACM": "sheafnc_checkpoints/m48c6j5w/GeneralSheaf-ACM-epoch=126.ckpt",
    "GeneralSheaf-DBLP": "sheafnc_checkpoints/92avtmkt/GeneralSheaf-DBLP-epoch=153.ckpt"
}


@hydra.main(version_base="1.2", config_path="configs", config_name="sheaf_config")
def main(cfg: Config) -> None:
    torch.set_float32_matmul_precision("high")

    # 1) get the datamodule
    datamodule = get_dataset_nc(cfg.dataset.name, homogeneous=True)
    datamodule.prepare_data()
    data: Data = datamodule.pyg_datamodule.data

    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    cfg.model_args.graph_size = datamodule.graph_size
    cfg.model_args.input_dim = datamodule.in_channels
    cfg.model_args.output_dim = datamodule.num_classes
    edge_index = datamodule.edge_index.to(cfg.model_args.device)

    model_cls = get_sheaf_model(cfg.model.type)
    encoder = model_cls(edge_index, cfg.model_args)

    model = SheafNodeClassifier.load_from_checkpoint(
        checkpoint_path=checkpoint_paths[f"{cfg.model.type}-{cfg.dataset.name}"],
        model=encoder
    )

    # 3) calculate the restriction maps
    encoder: SheafDiffusion = model.encoder
    x = data.x.to(cfg.model_args.device)
    x = F.dropout(x, p=encoder.input_dropout, training=encoder.training)
    x = encoder.lin1(x)
    if encoder.use_act:
        x = F.elu(x)
    x = F.dropout(x, p=encoder.dropout, training=encoder.training)
    if encoder.second_linear:
        x = encoder.lin12(x)
    x = x.view(encoder.graph_size * encoder.final_d, -1)
    x_maps = F.dropout(x, 0, training=False)
    maps = encoder.sheaf_learners[0](x_maps.reshape(encoder.graph_size, -1),
                                     edge_index)

    # 4) calculate the singular values (only if not diagonal)
    if cfg.model.type == ModelTypes.BundleSheaf:
        transform = Orthogonal(encoder.d, cfg.model_args.orth)
        maps = transform(maps)

    if cfg.model.type != ModelTypes.DiagSheaf:
        singular_values = torch.linalg.svdvals(maps).detach().cpu().numpy()
    else:
        diag_sort, _ = torch.sort(torch.square(maps), descending=True)
        singular_values = diag_sort.cpu().detach().numpy()

    print(singular_values.shape)
    print(data.edge_type.shape)
    print(torch.square(maps)[0])
    print(singular_values[0])

    np.save(f"tsne-input/{cfg.model.type}-{cfg.dataset.name}.npy", singular_values)
    np.save(f"tsne-input/{cfg.model.type}-{cfg.dataset.name}-labels.npy",
            data.edge_type.cpu().detach().numpy())


if __name__ == '__main__':
    main()
