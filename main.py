import sys

import wandb
from omegaconf import OmegaConf as omg
import torch

from agent import Agent
from trainer import Trainer, reward_fn
from data import DataGenerator


def load_conf():
    """Quick method to load configuration (using OmegaConf). By default,
    configuration is loaded from the default config file (config.yaml).
    Another config file can be specific through command line.
    Also, configuration can be over-written by command line.

    Returns:
        OmegaConf.DictConfig: OmegaConf object representing the configuration.
    """
    default_conf = omg.create({"config" : "config.yaml"})

    sys.argv = [a.strip("-") for a in sys.argv]
    cli_conf = omg.from_cli()

    yaml_file = omg.merge(default_conf, cli_conf).config

    yaml_conf = omg.load(yaml_file)

    return omg.merge(default_conf, yaml_conf, cli_conf)


def main():
    conf = load_conf()
    wandb.init(project=conf.proj_name, config=dict(conf))

    agent = Agent(embed_hidden=conf.embed_hidden, enc_stacks=conf.enc_stacks, ff_hidden=conf.ff_hidden, enc_heads=conf.enc_heads, query_hidden=conf.query_hidden, att_hidden=conf.att_hidden, crit_hidden=conf.crit_hidden, n_history=conf.n_history, p_dropout=conf.p_dropout)
    wandb.watch(agent)

    dataset = DataGenerator()

    trainer = Trainer(conf, agent, dataset)
    trainer.run()

    # Save trained agent
    torch.save(agent.state_dict(), conf.model_path)

    if conf.test:
        device = torch.device(conf.device)
        # Load trained agent
        agent.load_state_dict(torch.load(conf.model_path))
        agent.eval()
        agent = agent.to(device)

        running_reward = 0
        for _ in range(conf.test_steps):
            input_batch = dataset.test_batch(conf.batch_size, conf.max_len, conf.dimension, shuffle=False)
            input_batch = torch.Tensor(input_batch).to(device)

            tour, *_ = agent(input_batch)

            reward = reward_fn(input_batch, tour)

            # Find best solution
            j = reward.argmin()
            best_tour = tour[j][:-1].tolist()

            # Log
            running_reward += reward[j]

            # Display
            print('Reward (before 2 opt)', reward[j])
            opt_tour, opt_length = dataset.loop2opt(input_batch.cpu()[0][best_tour])
            print('Reward (with 2 opt)', opt_length)
            dataset.visualize_2D_trip(opt_tour)

        wandb.run.summary["test_reward"] = running_reward / conf.test_steps


if __name__ == "__main__":
    main()
