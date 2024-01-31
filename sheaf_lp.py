from core.datasets import get_dataset_lp, LinkPredDatasets
import torch

def main():
    torch.set_float32_matmul_precision("high")
    datamodule = get_dataset_lp(LinkPredDatasets.LastFM)
    datamodule.prepare_data()

    data = datamodule.train_data.to_homogeneous()

    print(data.edge_label[~data.edge_label.isnan()].shape)


if __name__ == "__main__":
    main()