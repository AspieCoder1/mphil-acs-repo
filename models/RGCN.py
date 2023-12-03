import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data.lightning.datamodule import LightningNodeData
from torch_geometric.datasets import Entities
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import RGCNConv
from torchmetrics.classification import Accuracy


class RGCN(nn.Module):
    def __init__(self, hidden_channels: int, num_classes: int, num_nodes: int,
                 num_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(num_nodes, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.linear = nn.Linear(hidden_channels, num_classes)

    def forward(self, data: Data):
        edge_type, edge_index = data.edge_type, data.edge_index

        x = self.conv1(None, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.linear(x)
        return x


class RGCNEntityPredictor(L.LightningModule):
    def __init__(self, hidden_channels: int, num_classes: int, num_nodes: int,
                 num_relations: int):
        super().__init__()
        self.model = RGCN(hidden_channels, num_classes, num_nodes, num_relations)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, batch: Data, _batch_idx) -> STEP_OUTPUT:
        print(batch)
        y = batch.train_y
        print(y)
        y = F.one_hot(y, num_classes=4)
        y_hat = self.model(batch)[batch.train_x]

        loss = F.cross_entropy(y_hat, y)
        train_acc = self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    dataset = Entities("datasets", "MUTAG")
    data = dataset[0]

    print(data.edge_weight)

    datamodule = LightningNodeData(
        data,
        input_train_nodes=data.train_idx,
        num_neighbors=[10, 10],
        batch_size=1024,
        num_workers=2,
    )

    model = RGCNEntityPredictor(16, dataset.num_classes, data.num_nodes,
                                dataset.num_relations)

    trainer = L.Trainer(accelerator="cpu", fast_dev_run=True)
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader())


if __name__ == '__main__':
    main()
