import torch


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