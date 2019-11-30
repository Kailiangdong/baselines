import numpy as np
from mpi4py import MPI
from baselines.common import dataset

class Discriminator:
    def __init__(self, reward_giver, expert_dataset, d_optim, d_step, d_stepsize, nworkers):
        self.reward_giver = reward_giver
        self.expert_dataset = expert_dataset
        self.d_step = d_step
        self.d_stepsize = d_stepsize
        self.d_optim = d_optim
        self.nworkers = nworkers

    def update(self, ob, ac):
        """
        Updates discriminator with a policy's generated observations and actions.
        Parameters:
            ob (list): A trajectory of observations.
            ac (list): A trajectory of actions.
        Returns:
            d_losses (list): The losses for the discriminator.
        """
        ob_expert, ac_expert = self.expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // self.d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, ac_expert = self.expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for reward_giver
            if hasattr(self.reward_giver, "obs_rms"): self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = self.reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            self.d_optim.update(allmean(g, self.nworkers), self.d_stepsize)
            d_losses.append(newlosses)

        return d_losses


