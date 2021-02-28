import wandb
from omegaconf import OmegaConf as omg

from agent import Agent
from trainer import Trainer


def load_conf():
    """Quick method to load configuration (using OmegaConf). By default,
    configuration is loaded from the default config file (config.yaml).
    Another config file can be specific through command line.
    Also, configuration can be over-written by command line.

    Returns:
        OmegaConf.DictConfig: OmegaConf object representing the configuration.
    """
    default_conf = omg.create({"config" : "config.yaml"})
    cli_conf = omg.from_cli()

    yaml_file = omg.merge(default_conf, cli_conf).config

    yaml_conf = omg.load(yaml_file)

    return omg.merge(default_conf, yaml_conf, cli_conf)


def main():
    conf = load_conf()
    wandb.init(project=conf.proj_name, config=dict(conf))

    agent = Agent(embed_hidden=conf.embed_hidden, enc_stacks=conf.enc_stacks, ff_hidden=conf.ff_hidden, enc_heads=conf.enc_heads, query_hidden=conf.query_hidden, att_hidden=conf.att_hidden, crit_hidden=conf.crit_hidden, n_history=conf.n_history, p_dropout=conf.p_dropout)
    wandb.watch(agent)

    trainer = Trainer(conf, agent)
    trainer.run()

if __name__ == "__main__":
    main()
