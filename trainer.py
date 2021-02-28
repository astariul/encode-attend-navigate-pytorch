import torch
from omegaconf import OmegaConf as omg


def reward(coords, tour):
    """Reward function. Compute the total distance for a tour, given the
    coordinates of each city and the tour indexes.

    Args:
        coords (torch.Tensor): Tensor of size [batch_size, seq_len, dim],
            representing each city's coordinates.
        tour (torch.Tensor): Tensor of size [batch_size, seq_len + 1],
            representing the tour's indexes (comes back to the first city).

    Returns:
        float: Reward for this tour.
    """
    dim = coords.size(-1)

    ordered_coords = torch.gather(coords, 1, tour.long().unsqueeze(-1).repeat(1, 1, dim))
    ordered_coords = ordered_coords.transpose(0, 2)   # [dim, seq_len, batch_size]

    # For each dimension (x, y), compute the squared difference between each city
    delta2 = [torch.square(d[1:] - d[:-1]).transpose(0, 1) for d in ordered_coords]

    # Euclidian distance between each city
    inter_city_distances = torch.sqrt(sum(delta2))
    distance = inter_city_distances.sum(dim=-1)
    
    return distance.float()


class Trainer():
    def __init__(self):
        conf = load_conf()
        # Have to init wandb with this config


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