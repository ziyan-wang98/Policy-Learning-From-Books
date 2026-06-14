import dataclasses
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
import numpy as np
from ....models.torch import (
    ActionOutput,
    CategoricalPolicy,
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    NormalPolicy,
    Parameter,
    Policy,
    build_squashed_gaussian_distribution,
    get_parameter,
)
from ....torch_utility import Modules, TorchMiniBatch, hard_sync, soft_sync
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase
from .ddpg_impl import DDPGBaseActorLoss, DDPGBaseImpl, DDPGBaseModules
from .utility import DiscreteQFunctionMixin

__all__ = [
    "SACImpl",
    "DiscreteSACImpl",
    "DiscreteCQLSACImpl",
    "SACModules",
    "DiscreteSACModules",
    "DiscreteCQLSACModules",
    "SACActorLoss",
]


@dataclasses.dataclass(frozen=True)
class SACModules(DDPGBaseModules):
    policy: NormalPolicy
    log_temp: Parameter
    temp_optim: Optional[Optimizer]


@dataclasses.dataclass(frozen=True)
class SACActorLoss(DDPGBaseActorLoss):
    temp: torch.Tensor
    temp_loss: torch.Tensor


class SACImpl(DDPGBaseImpl):
    _modules: SACModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: SACModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            device=device,
        )

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> SACActorLoss:
        dist = build_squashed_gaussian_distribution(action)
        sampled_action, log_prob = dist.sample_with_log_prob()

        if self._modules.temp_optim:
            temp_loss = self.update_temp(log_prob)
        else:
            temp_loss = torch.tensor(
                0.0, dtype=torch.float32, device=sampled_action.device
            )

        entropy = get_parameter(self._modules.log_temp).exp() * log_prob
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, sampled_action, "min"
        )
        return SACActorLoss(
            actor_loss=(entropy - q_t).mean(),
            temp_loss=temp_loss,
            temp=get_parameter(self._modules.log_temp).exp(),
        )

    def update_temp(self, log_prob: torch.Tensor) -> torch.Tensor:
        assert self._modules.temp_optim
        self._modules.temp_optim.zero_grad()
        with torch.no_grad():
            targ_temp = log_prob - self._action_size
        loss = -(get_parameter(self._modules.log_temp).exp() * targ_temp).mean()
        loss.backward()
        self._modules.temp_optim.step()
        return loss

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = build_squashed_gaussian_distribution(
                self._modules.policy(batch.next_observations)
            )
            action, log_prob = dist.sample_with_log_prob()
            entropy = get_parameter(self._modules.log_temp).exp() * log_prob
            target = self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = build_squashed_gaussian_distribution(self._modules.policy(x))
        return dist.sample()


@dataclasses.dataclass(frozen=True)
class DiscreteSACModules(Modules):
    policy: CategoricalPolicy
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    log_temp: Optional[Parameter]
    actor_optim: Optimizer
    critic_optim: Optimizer
    temp_optim: Optional[Optimizer]



@dataclasses.dataclass(frozen=True)
class DiscreteCQLSACModules(DiscreteSACModules):
    log_alpha: Parameter
    alpha_optim: Optional[Optimizer]


class DiscreteSACImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _modules: DiscreteSACModules
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _target_update_interval: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteSACModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        target_update_interval: int,
        ent_target_coef: float,
        gamma: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._gamma = gamma
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._target_update_interval = target_update_interval
        self._ent_target_coef = ent_target_coef
        hard_sync(modules.targ_q_funcs, modules.q_funcs)

    def update_critic(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._modules.critic_optim.step()

        return {"critic_loss": float(loss.cpu().detach().numpy())}

    def compute_q_pi(self, obs: TorchObservation, consider_entropy=False, reduction: str = "min") -> torch.Tensor:
        dist = self._modules.policy(obs)
        log_probs = dist.logits
        probs = dist.probs
        if self._modules.log_temp is None:
            temp = torch.zeros_like(log_probs)
        else:
            temp = get_parameter(self._modules.log_temp).exp()
        entropy = temp * log_probs
        target = self._q_func_forwarder.compute_target(obs, reduction=reduction)
        if target.dim() == 3:
            entropy = entropy.unsqueeze(0)
            probs = probs.unsqueeze(0)
        if consider_entropy:
            return (probs * (target - entropy)).sum(dim=-1, keepdim=True)
        else:
            return (probs * target).sum(dim=-1, keepdim=True)
    
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = self._modules.policy(batch.next_observations)
            log_probs = dist.logits
            probs = dist.probs
            if self._modules.log_temp is None:
                temp = torch.zeros_like(log_probs)
            else:
                temp = get_parameter(self._modules.log_temp).exp()
            entropy = temp * log_probs
            target = self._targ_q_func_forwarder.compute_target(
                batch.next_observations
            )
            keepdims = True
            if target.dim() == 3:
                entropy = entropy.unsqueeze(-1)
                probs = probs.unsqueeze(-1)
                keepdims = False
            return (probs * (target - entropy)).sum(dim=1, keepdim=keepdims)

    def compute_critic_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def update_actor(self, batch: TorchMiniBatch) -> Dict[str, float]:
        # Q function should be inference mode for stability
        self._modules.q_funcs.eval()

        self._modules.actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._modules.actor_optim.step()

        return {"actor_loss": float(loss.cpu().detach().numpy())}

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            q_t = self._q_func_forwarder.compute_expected_q(
                batch.observations, reduction="min"
            )
        dist = self._modules.policy(batch.observations)
        log_probs = dist.logits
        probs = dist.probs
        if self._modules.log_temp is None:
            temp = torch.zeros_like(log_probs)
        else:
            temp = get_parameter(self._modules.log_temp).exp()
        entropy = temp * log_probs
        return (probs * (entropy - q_t)).sum(dim=1).mean()

    def update_temp(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._modules.temp_optim
        assert self._modules.log_temp is not None
        self._modules.temp_optim.zero_grad()

        with torch.no_grad():
            dist = self._modules.policy(batch.observations)
            log_probs = F.log_softmax(dist.logits, dim=1)
            probs = dist.probs
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdim=True)
            entropy_target = self._ent_target_coef * (-math.log(1 / self.action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(get_parameter(self._modules.log_temp).exp() * targ_temp).mean()

        loss.backward()
        self._modules.temp_optim.step()

        # current temperature value
        log_temp = get_parameter(self._modules.log_temp)
        cur_temp = log_temp.exp().cpu().detach().numpy()[0][0]

        return {
            "temp/loss": float(loss.cpu().detach().numpy()),
            "temp/v": float(cur_temp),
            "temp/expct_log_probs": float(expct_log_probs.mean().cpu().detach().numpy()),
        }

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}

        # lagrangian parameter update for SAC temeprature
        if self._modules.temp_optim:
            metrics.update(self.update_temp(batch))
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        soft_sync(self._modules.targ_q_funcs, self._modules.q_funcs, 0.005)
        # if grad_step % self._target_update_interval == 0:
        #     self.update_target()

        return metrics

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        dist = self._modules.policy(x)
        return dist.probs.argmax(dim=1)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = self._modules.policy(x)
        return dist.sample()

    def update_target(self) -> None:
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    @property
    def policy(self) -> Policy:
        return self._modules.policy

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.actor_optim

    @property
    def q_function(self) -> nn.ModuleList:
        return self._modules.q_funcs

    @property
    def q_function_optim(self) -> Optimizer:
        return self._modules.critic_optim

@dataclasses.dataclass(frozen=True)
class CQLSACCriticLoss:
    td_loss: torch.Tensor
    critic_loss: torch.Tensor
    raw_conservative_loss: torch.Tensor
    conservative_loss: torch.Tensor
    alpha: torch.Tensor
    logsumexp: torch.Tensor
    data_values: torch.Tensor

class DiscreteCQLSACImpl(DiscreteSACImpl):
    """
    The implementation refer to d3rlpy.algos.qlearning.torch.cql_sac_impl.CQLImpl and d3rlpy.algos.qlearning.torch.cql_sac_impl.DiscreteCQLImpl
    """
    _modules: DiscreteCQLSACModules
    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteCQLSACModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        target_update_interval: int,
        gamma: float,
        device: str,
        alpha_threshold: float = 5.0,
        conservative_weight: float = 5.0,
        n_action_samples: int = 10,
        soft_q_backup: bool = False,
        ent_target_coef: float = 0.98,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=target_update_interval,
            gamma=gamma,
            ent_target_coef=ent_target_coef,
            device=device)
        self._alpha_threshold = alpha_threshold
        if conservative_weight < 0:
            self._conservative_weight = 1.0
        else:
            self._conservative_weight = conservative_weight
        self._n_action_samples = n_action_samples
        self._soft_q_backup = soft_q_backup

    def update_alpha(self, conservative_loss: torch.Tensor) -> None:
        assert self._modules.alpha_optim
        self._modules.alpha_optim.zero_grad()
        # the original implementation does scale the loss value
        loss = -conservative_loss
        loss.backward()
        self._modules.alpha_optim.step()

    def _compute_conservative_loss(
        self,
        obs_t: TorchObservation,
        act_t: torch.Tensor,
        obs_tp1: TorchObservation,
    ) -> torch.Tensor:
        # Implement different variants.
        # 1. Compute logsumexp using policy-based importance sampling.
        logsumexp = self.compute_q_pi(obs_t, reduction="none").mean(dim=0).mean()
        logsumexp = torch.clamp(logsumexp, 0, 1e6)
        # 2. Compute logsumexp using random-policy importance sampling.
        pass
        # policy_values_t = self._compute_policy_is_values(obs_t, obs_t)
        # policy_values_tp1 = self._compute_policy_is_values(obs_tp1, obs_t)
        # random_values = self._compute_random_is_values(obs_t)

        # # compute logsumexp
        # # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        # target_values = torch.cat(
        #     [policy_values_t, policy_values_tp1, random_values], dim=2
        # )
        # logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # # estimate action-values for data actions
        one_hot = F.one_hot(act_t.view(-1).long(), num_classes=self.action_size)
        data_values = self._q_func_forwarder.compute_expected_q(obs_t, "none")
        data_values = (data_values * one_hot).sum(dim=2, keepdim=True).mean(dim=0).mean()
        data_values = torch.clamp(data_values, None, 100)
        loss = logsumexp - data_values
        scaled_loss = loss

        # # clip for stability
        log_alpha = get_parameter(self._modules.log_alpha)
        clipped_alpha = log_alpha.exp().clamp(0, 1e6)[0][0]

        if torch.abs(scaled_loss - self._alpha_threshold).item() <  np.maximum(5e-3, self._alpha_threshold/10):
            clipped_alpha *= 0.0
        return clipped_alpha * self._conservative_weight * torch.square(scaled_loss - self._alpha_threshold), scaled_loss, logsumexp, data_values

    def compute_critic_loss(self, batch: TorchMiniBatch, q_tpn: torch.Tensor) -> CQLSACCriticLoss:
        loss = super().compute_critic_loss(batch, q_tpn)
        conservative_loss, raw_conservative_loss, logsumexp, data_values = self._compute_conservative_loss(
            batch.observations, batch.actions.long(), batch.next_observations
        )
        return CQLSACCriticLoss(
            critic_loss=loss + conservative_loss,
            td_loss = loss,
            conservative_loss=conservative_loss,
            raw_conservative_loss=raw_conservative_loss,
            logsumexp=logsumexp,
            data_values=data_values,
            alpha=get_parameter(self._modules.log_alpha).exp(),
        )

    def update_critic(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.critic_optim.zero_grad()
        # self._modules.alpha_optim.zero_grad()
        q_tpn = self.compute_target(batch)
        q_tpn = torch.clamp(q_tpn, -1/(1-self._gamma), 1/(1-self._gamma))
        all_loss = self.compute_critic_loss(batch, q_tpn)
        loss = all_loss.critic_loss
            # self.update_alpha(all_loss.conservative_loss)
        loss.backward()
        self._modules.critic_optim.step()
        if self._modules.alpha_optim and all_loss.conservative_loss.item() != 0:
            # the original implementation does scale the loss value
            alpha_loss = self._compute_conservative_loss(batch.observations, batch.actions.long(), batch.next_observations)[0]
            self.update_alpha(alpha_loss)

        return {"td_loss": float(all_loss.td_loss.cpu().detach().numpy()),
                "qtarget/mean": float(q_tpn.mean().cpu().detach().numpy()),
                "alpha": float(all_loss.alpha.cpu().detach().numpy()),
                "raw_con_loss": float(all_loss.raw_conservative_loss.cpu().detach().numpy()),
                "logsumexp": float(all_loss.logsumexp.cpu().detach().numpy()),
                "data_values": float(all_loss.data_values.cpu().detach().numpy()),
                "critic_loss": float(loss.cpu().detach().numpy()), 
                "con_loss": float(all_loss.conservative_loss.cpu().detach().numpy()),
                "qtarget/std": float(q_tpn.std().cpu().detach().numpy()),
                "qtarget/max": float(q_tpn.max().cpu().detach().numpy()),
                "qtarget/min": float(q_tpn.min().cpu().detach().numpy()),}
    
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        if self._soft_q_backup:
            target_value = super().compute_target(batch)
        else:
            target_value = self._compute_deterministic_target(batch)
        return target_value

    def _compute_deterministic_target(
        self, batch: TorchMiniBatch
    ) -> torch.Tensor:
        with torch.no_grad():
            action = self.inner_predict_best_action(batch.next_observations)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
        
        