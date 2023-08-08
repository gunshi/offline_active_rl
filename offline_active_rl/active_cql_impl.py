import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from d3rlpy.dataset import Shape
from d3rlpy.models.torch import EnsembleDiscreteQFunction
from d3rlpy.torch_utility import TorchMiniBatch, train_api
from d3rlpy.algos.qlearning.torch.dqn_impl import DoubleDQNImpl


__all__ = ["DiscreteActiveCQLImpl"]


class DiscreteActiveCQLImpl(DoubleDQNImpl):
    _alpha: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleDiscreteQFunction,
        optim: Optimizer,
        gamma: float,
        alpha: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            q_func=q_func,
            optim=optim,
            gamma=gamma,
            device=device,
        )
        self._alpha = alpha

    def compute_indep_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:

        td_sum = torch.tensor(
            0.0, dtype=torch.float32, device=observations.device
        )
        for q_itr, q_func in enumerate(self._q_func._q_funcs):
            loss = q_func.compute_error(
                observations=observations,
                actions=actions,
                rewards=rewards,
                target=target[q_itr],
                terminals=terminals,
                gamma=gamma,
                reduction="none",
            )
            td_sum += loss.mean()
        return td_sum
    
    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        
        loss = self.compute_indep_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions.long()
        )
        return loss + self._alpha * conservative_loss, conservative_loss

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        # compute logsumexp
        policy_values = self._q_func(obs_t)
        logsumexp = torch.logsumexp(policy_values, dim=1, keepdim=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        data_values = (self._q_func(obs_t) * one_hot).sum(dim=1, keepdim=True)

        return (logsumexp - data_values).mean()

    @train_api
    def update(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._optim is not None

        self._optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss, cql_loss = self.compute_loss(batch, q_tpn)

        loss.backward()
        self._optim.step()

        return np.array(
            [loss.cpu().detach().numpy(), cql_loss.cpu().detach().numpy()]
        )
    
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self.inner_predict_best_action(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="none",
            )