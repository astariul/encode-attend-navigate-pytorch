import wandb
import torch
import torch.nn.functional as F


def reward_fn(coords, tour):
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
    def __init__(self, conf, agent, dataset):
        """Trainer class, taking care of training the agent.

        Args:
            conf (OmegaConf.DictConf): Configuration.
            agent (torch.nn.Module): Agent network to train.
            dataset (data.DataGenerator): Data generator.
        """
        super().__init__()

        self.conf = conf
        self.agent = agent
        self.dataset = dataset

        self.device = torch.device(self.conf.device)
        self.agent = self.agent.to(self.device)

        self.optim = torch.optim.Adam(params=self.agent.parameters(), lr=self.conf.lr)
        gamma = 1 - self.conf.lr_decay_rate / self.conf.lr_decay_steps      # To have same behavior as Tensorflow implementation
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optim, gamma=gamma)


    def train_step(self, data):
        self.optim.zero_grad()

        # Forward pass
        tour, critique, log_probs, _ = self.agent(data)

        # Compute reward
        reward = reward_fn(data, tour)

        # Compute losses for both actor (reinforce) and critic
        loss1 = ((reward - critique) * log_probs).mean()
        loss2 = F.mse_loss(reward, critique)

        # Backward pass
        loss1.backward()
        loss2.backward()

        # Optimize
        self.optim.step()

        # Update LR
        self.scheduler.step()

        return reward.mean(), [loss1, loss2]

    def run(self):
        self.agent.train()
        running_reward, running_losses = 0, [0, 0]
        for step in range(self.conf.steps):
            input_batch = self.dataset.train_batch(self.conf.batch_size, self.conf.max_len, self.conf.dimension)
            input_batch = torch.Tensor(input_batch).to(self.device)

            reward, losses = self.train_step(input_batch)

            running_reward += reward
            running_losses[0] += losses[0]
            running_losses[1] += losses[1]

            if step % self.conf.log_interval == self.conf.log_interval - 1:
                # Log stuff
                wandb.log({
                    'reward': running_reward / self.conf.log_interval,
                    'actor_loss': running_losses[0] / self.conf.log_interval,
                    'critic_loss': running_losses[1] / self.conf.log_interval,
                    'custom_step': step
                })

                # Reset running reward/loss
                running_reward, running_losses = 0, [0, 0]


            
