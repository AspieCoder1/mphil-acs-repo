#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

from dataclasses import dataclass


# parser = argparse.ArgumentParser()
# parser.add_argument("--train_prop", type=float, default=0.5)
# parser.add_argument("--valid_prop", type=float, default=0.25)
# parser.add_argument("--dname", default="walmart-trips-100")
# # method in ['SetGNN','CEGCN','CEGAT','HyperGCN','HGNN','HCHA']
# parser.add_argument("--method", default="AllSetTransformer")
# parser.add_argument("--epochs", default=500, type=int)
# # Number of runs for each split (test fix, only shuffle train/val)
# parser.add_argument("--runs", default=10, type=int)
# parser.add_argument("--cuda", default=0, choices=[-1, 0, 1], type=int)
# parser.add_argument("--dropout", default=0.5, type=float)
# parser.add_argument("--lr", default=0.001, type=float)
# parser.add_argument("--wd", default=0.0, type=float)
# # How many layers of full NLConvs
# parser.add_argument("--All_num_layers", default=2, type=int)
# parser.add_argument(
#     "--MLP_num_layers", default=2, type=int
# )  # How many layers of encoder
# parser.add_argument("--MLP_hidden", default=64, type=int)  # Encoder hidden units
# parser.add_argument(
#     "--Classifier_num_layers", default=2, type=int
# )  # How many layers of decoder
# parser.add_argument("--Classifier_hidden", default=64, type=int)  # Decoder hidden units
# parser.add_argument("--display_step", type=int, default=-1)
# parser.add_argument("--aggregate", default="mean", choices=["sum", "mean"])
# # ['all_one','deg_half_sym']
# parser.add_argument("--normtype", default="all_one")
# parser.add_argument("--add_self_loop", action="store_false")
# # NormLayer for MLP. ['bn','ln','None']
# parser.add_argument("--normalization", default="ln")
# parser.add_argument("--deepset_input_norm", default=True)
# parser.add_argument("--GPR", action="store_false")  # skip all but last dec
# # skip all but last dec
# parser.add_argument("--LearnMask", action="store_false")
# parser.add_argument("--num_features", default=0, type=int)  # Placeholder
# parser.add_argument("--num_classes", default=0, type=int)  # Placeholder
# # Choose std for synthetic feature noise
# parser.add_argument("--feature_noise", default="1", type=str)
# # whether the he contain self node or not
# parser.add_argument("--exclude_self", action="store_true")
# parser.add_argument("--PMA", action="store_true")
# #     Args for HyperGCN
# parser.add_argument("--HyperGCN_mediators", action="store_true")
# parser.add_argument("--HyperGCN_fast", default=True, type=bool)
# #     Args for Attentions: GAT and SetGNN
# parser.add_argument("--heads", default=1, type=int)  # Placeholder
# parser.add_argument("--output_heads", default=1, type=int)  # Placeholder
# #     Args for HNHN
# parser.add_argument("--HNHN_alpha", default=-1.5, type=float)
# parser.add_argument("--HNHN_beta", default=-0.5, type=float)
# parser.add_argument("--HNHN_nonlinear_inbetween", default=True, type=bool)
# #     Args for HCHA
# parser.add_argument("--HCHA_symdegnorm", action="store_true")
# #     Args for UniGNN
# parser.add_argument(
#     "--UniGNN_use-norm", action="store_true", help="use norm in the final layer"
# )
# parser.add_argument("--UniGNN_degV", default=0)
# parser.add_argument("--UniGNN_degE", default=0)
# parser.add_argument("--wandb", default=True, type=str2bool)
# parser.add_argument("--activation", default="relu", choices=["Id", "relu", "prelu"])
#
# # # Args just for EDGNN
# # parser.add_argument('--MLP2_num_layers', default=-1, type=int, help='layer number of mlp2')
# # parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3')
# # parser.add_argument('--edconv_type', default='EquivSet', type=str, choices=['EquivSet', 'JumpLink', 'MeanDeg', 'Attn', 'TwoSets'])
# # parser.add_argument('--restart_alpha', default=0.5, type=float)
#
# # Args for Sheaves
# parser.add_argument("--init_hedge", default="rand", type=str, choices=["rand", "avg"])
# parser.add_argument(
#     "--use_attention", type=str2bool, default=True
# )  # used in HCHA. if true ypergraph attention otherwise hypergraph conv
# parser.add_argument(
#     "--tag", type=str, default="testing"
# )  # helper for wandb in order to filter out the testing runs. if set to testing we are in dev mode
# parser.add_argument(
#     "--sheaf_normtype",
#     type=str,
#     default="degree_norm",
#     choices=["degree_norm", "block_norm", "sym_degree_norm", "sym_block_norm"],
# )  # used to normalise the sheaf laplacian. will add other normalisations later
# parser.add_argument(
#     "--sheaf_act", type=str, default="sigmoid", choices=["sigmoid", "tanh", "none"]
# )  # final activation used after predicting the dxd block
# parser.add_argument(
#     "--sheaf_dropout", type=str2bool, default=False
# )  # final activation used after predicting the dxd block
# parser.add_argument(
#     "--sheaf_left_proj", type=str2bool, default=False
# )  # multiply to the left with IxW
# parser.add_argument(
#     "--dynamic_sheaf", type=str2bool, default=False
# )  # if set to True, a different sheaf is predicted at each layer
# parser.add_argument(
#     "--sheaf_special_head", type=str2bool, default=False
# )  # if set to True, a special head corresponding to alpha=1 and d=heads-1 in that case)
# parser.add_argument(
#     "--sheaf_pred_block", type=str, default="MLP_var1"
# )  # if set to True, a special head corresponding to alpha=1 and d=heads-1 in that case)
# parser.add_argument(
#     "--sheaf_transformer_head", type=int, default=1
# )  # only when sheaf_pred_block==transformer. The number of transformer head used to predict the dxd blocks
# parser.add_argument("--AllSet_input_norm", default=True)
# parser.add_argument(
#     "--residual_HCHA", default=False
# )  # HCHA and *Sheafs only; if HCHA-based architectures have conv layers with residual connections
# parser.add_argument(
#     "--rank", default=0, type=int, help="rank for dxd blocks in LowRankSheafs"
# )  # ronly for ank for the low-rank matrix generation of the dxd block                                                                                          # should be < d
#
# parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
# parser.set_defaults(add_self_loop=True)
# parser.set_defaults(exclude_self=False)
# parser.set_defaults(GPR=False)
# parser.set_defaults(LearnMask=False)
# parser.set_defaults(HyperGCN_mediators=True)
# parser.set_defaults(HCHA_symdegnorm=False)


@dataclass
class HypergraphConfig:
    train_prop: float
    valid_prop: float
    dname: str
