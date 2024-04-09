#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from datasets.link_pred import (
    MovieLensDatamodule,
    LastFMDataModule,
    AmazonBooksDataModule,
)

if __name__ == "__main__":
    dm = LastFMDataModule(is_homogeneous=False)
    dm.prepare_data()
    print(f"{dm.num_nodes=}")
    print(f"{dm.num_node_types=}")
    print(f"{dm.num_edges=}")
    print(f"{dm.num_edge_types=}")
