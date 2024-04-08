#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from enum import auto

from strenum import UppercaseStrEnum, PascalCaseStrEnum

from datasets.hgb import (
    HGBBaseDataModule,
    DBLPDataModule,
    ACMDataModule,
    IMDBDataModule,
)
from datasets.hgt import (
    HGTBaseDataModule,
    HGTDBLPDataModule,
    HGTACMDataModule,
    HGTIMDBDataModule,
)
from datasets.link_pred import (
    LinkPredBase,
    LastFMDataModule,
    AmazonBooksDataModule,
    MovieLensDatamodule,
)


class NCDatasets(UppercaseStrEnum):
    DBLP = auto()
    ACM = auto()
    IMDB = auto()


def get_dataset_nc(dataset: NCDatasets, homogeneous: bool = False) -> HGBBaseDataModule:
    if dataset == NCDatasets.DBLP:
        return DBLPDataModule(homogeneous=homogeneous)
    elif dataset == NCDatasets.ACM:
        return ACMDataModule(homogeneous=homogeneous)
    else:
        return IMDBDataModule(homogeneous=homogeneous)


def get_dataset_hgt(dataset: NCDatasets) -> HGTBaseDataModule:
    if dataset == NCDatasets.DBLP:
        return HGTDBLPDataModule()
    elif dataset == NCDatasets.ACM:
        return HGTACMDataModule()
    else:
        return HGTIMDBDataModule()


class LinkPredDatasets(PascalCaseStrEnum):
    LastFM = "LastFM"
    AmazonBooks = auto()
    MovieLens = auto()


def get_dataset_lp(
    dataset: LinkPredDatasets, is_homogeneous: bool = False
) -> LinkPredBase:
    if dataset == LinkPredDatasets.LastFM:
        return LastFMDataModule(is_homogeneous=is_homogeneous)
    elif dataset == LinkPredDatasets.AmazonBooks:
        return AmazonBooksDataModule(is_homogeneous=is_homogeneous)
    else:
        return MovieLensDatamodule(is_homogeneous=is_homogeneous)
