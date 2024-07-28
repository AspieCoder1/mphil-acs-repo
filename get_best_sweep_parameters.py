#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT
import click
import wandb
import yaml


def get_best_parameters(sweep_id: str) -> str:
    api = wandb.Api()
    sweep = api.sweep(f'acs-thesis-lb2027/hetero-sheaf-paper/{sweep_id}')
    best_run = sweep.best_run(order='val/micro-f1')
    best_parameters = best_run.config
    print(best_parameters)
    return best_parameters


def main():
    names = {
        "dblp": "udl9xl76",
        "acm": "innurdsp",
        "imdb": "phsd9cm9",
        "pubmed_nc": "4sl4f8sx",
    }
    for dataset, sweep_id in names.items():
        click.secho(f"Processing {dataset}", bold=True)
        best_params = get_best_parameters(sweep_id)
        # with open(f"configs/experiment/sheaf_nc/{dataset}.yaml", 'w+') as f:
        #     yaml.dump(best_params, f, default_flow_style=False)


if __name__ == '__main__':
    main()
