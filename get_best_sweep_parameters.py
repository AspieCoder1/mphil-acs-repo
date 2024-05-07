#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import wandb


def main(sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f'acs-thesis-lb2027/gnn-baselines/{sweep_id}')
    best_run = sweep.best_run(order='val/accuracy')
    best_parameters = best_run.config
    print(best_parameters)


if __name__ == '__main__':
    main('f0ivn682')
