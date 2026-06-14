import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Dict, Union, Optional, List

import torch
from torch.optim import Optimizer

from ....models.torch import (
    CategoricalPolicy,
    DeterministicPolicy,
    NormalPolicy,
    Policy,
    EnsemblePiFunctionForwarder,
    compute_deterministic_imitation_loss,
    compute_discrete_imitation_loss,
    compute_stochastic_imitation_loss,
    compute_ensemble_stochastic_imitation_loss,
    compute_stochastic_weighted_imitation_loss,
)
from ....torch_utility import Modules, TorchMiniBatch
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase

__all__ = ["BCImpl", "DiscreteBCImpl", "BCModules", "DiscreteBCModules"]


@dataclasses.dataclass(frozen=True)
class BCBaseModules(Modules):
    optim: Optimizer


class BCBaseImpl(QLearningAlgoImplBase, metaclass=ABCMeta):
    _modules: BCBaseModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BCBaseModules,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )

    def update_imitator(self, batch: TorchMiniBatch) -> float:
        self._modules.optim.zero_grad()

        loss = self.compute_loss(batch.observations, batch.actions)

        loss.backward()
        self._modules.optim.step()

        return float(loss.cpu().detach().numpy())

    @abstractmethod
    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        pass

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("BC does not support value estimation")

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        return {"loss": self.update_imitator(batch)}


@dataclasses.dataclass(frozen=True)
class BCModules(BCBaseModules):
    imitator: Union[DeterministicPolicy, NormalPolicy]


class BCImpl(BCBaseImpl):
    _modules: BCModules
    _policy_type: str

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BCModules,
        policy_type: str,
        device: str,
        action_min: Optional[list],
        action_max: Optional[list],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._policy_type = policy_type
        # useless for now, d3rlpy already implemented this in base class.
        if action_min is not None and action_max is not None:
            self.action_min = torch.tensor(action_min).to(device)
            self.action_max = torch.tensor(action_max).to(device)
            self.action_shift = (self.action_max + self.action_min) / 2
            self.action_scale = (self.action_max - self.action_min) / 2
        else:
            self.action_shift =  torch.tensor(0).to(device)
            self.action_scale = torch.tensor(1).to(device)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        res = self._modules.imitator(x)
        norm_acs = torch.normal(res.squashed_mu, torch.exp(res.logstd))
        return norm_acs * self.action_scale + self.action_shift
    
    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        res = self._modules.imitator(x)
        norm_acs = res.squashed_mu
        return norm_acs * self.action_scale + self.action_shift

    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        
        norm_act_t = (act_t - self.action_shift) / self.action_scale
        if self._policy_type == "deterministic":
            return compute_deterministic_imitation_loss(
                self._modules.imitator, obs_t, norm_act_t
            )
        elif self._policy_type == "stochastic":
            return compute_stochastic_imitation_loss(
                self._modules.imitator, obs_t, norm_act_t
            )
        elif self._policy_type == "stochastic_weighted" or self._policy_type == 'stochastic_weighted_multi_head':
            return compute_stochastic_weighted_imitation_loss(
                self._modules.imitator, obs_t, norm_act_t
            )
        else:
            raise ValueError(f"invalid policy_type: {self._policy_type}")

    @property
    def policy(self) -> Policy:
        return self._modules.imitator

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.optim


@dataclasses.dataclass(frozen=True)
class EnsembleBCModules(BCBaseModules):
    imitator: List[NormalPolicy]



class EnsembleBCImpl(BCImpl):
    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: EnsembleBCModules,
        device: str,
        imitator_forwarder: EnsemblePiFunctionForwarder,
        action_min: Optional[list],
        action_max: Optional[list],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            policy_type='stochastic',
            device=device,
            action_min=action_min,
            action_max=action_max)
        self._imitator_forwarder = imitator_forwarder

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        norm_acs = self._imitator_forwarder.sample_action(x)
        return norm_acs * self.action_scale + self.action_shift
    
    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        norm_acs = self._imitator_forwarder.best_action(x)
        return norm_acs * self.action_scale + self.action_shift

    def inner_predict_uncertainty(self, x: TorchObservation) -> torch.Tensor:
        return self._imitator_forwarder.uncertainty(x) * self.action_scale + self.action_shift
    
    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        norm_act_t = (act_t - self.action_shift) / self.action_scale
        return compute_ensemble_stochastic_imitation_loss(self._modules.imitator, obs_t, norm_act_t)
    

    @property
    def policy(self) -> Policy:
        return self._modules.imitator

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.optim

@dataclasses.dataclass(frozen=True)
class DiscreteBCModules(BCBaseModules):
    imitator: CategoricalPolicy


class DiscreteBCImpl(BCBaseImpl):
    _modules: DiscreteBCModules
    _beta: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteBCModules,
        beta: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._beta = beta

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._modules.imitator(x).logits.argmax(dim=1)

    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        return compute_discrete_imitation_loss(
            policy=self._modules.imitator,
            x=obs_t,
            action=act_t.long(),
            beta=self._beta,
        )
