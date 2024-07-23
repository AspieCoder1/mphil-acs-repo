#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
from typing import Type, Union

import torch
from torch_geometric.data import Data, HeteroData

from datasets.hgb import (PubMedDataModule, ACMDataModule, HGBBaseDataModule,
                          DBLPDataModule, FreebaseDataModule, IMDBDataModule, )
from datasets.link_pred import PubMedLPDataModule, LastFMDataModule

DataOrHetero = Union[Data, HeteroData]


def generate_data_modules_nc_tuning_mode(datamodule: Type[HGBBaseDataModule]) -> [
    DataOrHetero,
    DataOrHetero,
    str]:
    dm1 = datamodule(data_dir="../data", hyperparam_tuning=True)
    dm2 = datamodule(data_dir="../data", hyperparam_tuning=True)
    dm1.prepare_data()
    dm2.prepare_data()

    return dm1.pyg_datamodule.data, dm2.pyg_datamodule.data, dm1.target


def generate_data_modules_nc(datamodule: Type[HGBBaseDataModule]) -> [
    DataOrHetero,
    DataOrHetero,
    str]:
    dm1 = datamodule(data_dir="../data")
    dm2 = datamodule(data_dir="../data")
    dm1.prepare_data()
    dm2.prepare_data()

    return dm1.pyg_datamodule.data, dm2.pyg_datamodule.data, dm1.target


def generate_data_modules_lp_tuning_mode(
        datamodule: Type[Union[PubMedLPDataModule, LastFMDataModule]]) -> [DataOrHetero,
                                                                           DataOrHetero,
                                                                           str]:
    dm1 = datamodule(data_dir="../data", hyperparam_tuning=True, homogeneous=False)
    dm2 = datamodule(data_dir="../data", hyperparam_tuning=True, homogeneous=False)
    dm1.prepare_data()
    dm2.prepare_data()

    return dm1.data, dm2.data, dm1.target


def test_pubmed_nc_tuning_mode():
    data1, data2, target = generate_data_modules_nc_tuning_mode(PubMedDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert torch.allclose(mask1, mask2)


def test_pubmed_nc():
    data1, data2, target = generate_data_modules_nc(PubMedDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert not torch.allclose(mask1, mask2)


def test_acm_tuning_mode():
    data1, data2, target = generate_data_modules_nc_tuning_mode(ACMDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert torch.allclose(mask1, mask2)


def test_acm_nc():
    data1, data2, target = generate_data_modules_nc(ACMDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert not torch.allclose(mask1, mask2)


def test_dblp_tuning_mode():
    data1, data2, target = generate_data_modules_nc_tuning_mode(DBLPDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert torch.allclose(mask1, mask2)


def test_freebase_tuning_mode():
    data1, data2, target = generate_data_modules_nc_tuning_mode(FreebaseDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert torch.allclose(mask1, mask2)


def test_freebase_nc():
    data1, data2, target = generate_data_modules_nc(FreebaseDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert not torch.allclose(mask1, mask2)


def test_imdb_tuning_mode():
    data1, data2, target = generate_data_modules_nc_tuning_mode(IMDBDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert torch.allclose(mask1, mask2)


def test_imdb_nc():
    data1, data2, target = generate_data_modules_nc(IMDBDataModule)

    mask1 = data1[target].val_mask
    mask2 = data2[target].val_mask
    assert not torch.allclose(mask1, mask2)


def test_pubmed_lp_tuning_mode():
    data1, data2, target = generate_data_modules_lp_tuning_mode(PubMedLPDataModule)

    edge_index1 = data1[target].val_edge_label_index
    edge_index2 = data2[target].val_edge_label_index
    assert torch.allclose(edge_index1, edge_index2)

    label1 = data1[target].val_edge_label
    label2 = data2[target].val_edge_label
    assert torch.allclose(label1, label2)


def test_lastfm_lp_tuning_mode():
    data1, data2, target = generate_data_modules_lp_tuning_mode(LastFMDataModule)

    edge_index1 = data1[target].val_edge_label_index
    edge_index2 = data2[target].val_edge_label_index
    assert torch.allclose(edge_index1, edge_index2)

    label1 = data1[target].val_edge_label
    label2 = data2[target].val_edge_label
    assert torch.allclose(label1, label2)
