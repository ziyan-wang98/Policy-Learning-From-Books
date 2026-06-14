from typing import Iterator, Optional, Sequence

import numpy as np
from typing_extensions import Protocol

from ..dataset import (
    EpisodeBase,
    ReplayBuffer,
    TransitionMiniBatch,
    TransitionPickerProtocol,
)
from typing import Union
from ..interface import QLearningAlgoProtocol
from ..types import GymEnv
from .utility import evaluate_qlearning_with_environment, evaluate_qlearning_with_img_obs_environment, evaluate_rule_based_with_environment, evaluate_llm_with_environment, evaluate_ttt_environment

__all__ = [
    "EvaluatorProtocol",
    "make_batches",
    "TDErrorEvaluator",
    "DiscountedSumOfAdvantageEvaluator",
    "AverageValueEstimationEvaluator",
    "InitialStateValueEstimationEvaluator",
    "SoftOPCEvaluator",
    "ContinuousActionDiffEvaluator",
    "DiscreteActionMatchEvaluator",
    "CompareContinuousActionDiffEvaluator",
    "CompareDiscreteActionMatchEvaluator",
    "EnvironmentEvaluator",
    "TTTEnvironmentEvaluator",
    "EnsembleDatasetErrorEvaluator",
]


WINDOW_SIZE = 1024


class EvaluatorProtocol(Protocol):
    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        """Computes metrics.

        Args:
            algo: Q-learning algorithm.
            dataset: ReplayBuffer.

        Returns:
            Computed metrics.
        """
        raise NotImplementedError


def make_batches(
    episode: EpisodeBase,
    window_size: int,
    transition_picker: TransitionPickerProtocol,
) -> Iterator[TransitionMiniBatch]:
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, episode.transition_count)
        transitions = [
            transition_picker(episode, index)
            for index in range(head_index, last_index)
        ]
        batch = TransitionMiniBatch.from_transitions(transitions)
        yield batch


class TDErrorEvaluator(EvaluatorProtocol):
    r"""Returns average TD error.

    This metric suggests how Q functions overfit to training sets.
    If the TD error is large, the Q functions are overfitting.

    .. math::

        \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D}
            [(Q_\theta (s_t, a_t)
             - r_{t+1} - \gamma \max_a Q_\theta (s_{t+1}, a))^2]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_errors = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                # estimate values for current observations
                values = algo.predict_value(batch.observations, batch.actions)

                # estimate values for next observations
                next_actions = algo.predict(batch.next_observations)
                next_values = algo.predict_value(
                    batch.next_observations, next_actions
                )

                # calculate td errors
                mask = (1.0 - batch.terminals).reshape(-1)
                rewards = np.asarray(batch.rewards).reshape(-1)
                if algo.reward_scaler:
                    rewards = algo.reward_scaler.transform_numpy(rewards)
                y = rewards + algo.gamma * next_values * mask
                total_errors += ((values - y) ** 2).tolist()

        return float(np.mean(total_errors))


class DiscountedSumOfAdvantageEvaluator(EvaluatorProtocol):
    r"""Returns average of discounted sum of advantage.

    This metric suggests how the greedy-policy selects different actions in
    action-value space.
    If the sum of advantage is small, the policy selects actions with larger
    estimated action-values.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D}
            [\sum_{t' = t} \gamma^{t' - t} A(s_{t'}, a_{t'})]

    where :math:`A(s_t, a_t) = Q_\theta (s_t, a_t)
    - \mathbb{E}_{a \sim \pi} [Q_\theta (s_t, a)]`.

    References:
        * `Murphy., A generalization error for Q-Learning.
          <http://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf>`_

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_sums = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                # estimate values for dataset actions
                dataset_values = algo.predict_value(
                    batch.observations, batch.actions
                )

                # estimate values for the current policy
                actions = algo.predict(batch.observations)
                on_policy_values = algo.predict_value(
                    batch.observations, actions
                )

                # calculate advantages
                advantages = (dataset_values - on_policy_values).tolist()

                # calculate discounted sum of advantages
                A = advantages[-1]
                sum_advantages = [A]
                for advantage in reversed(advantages[:-1]):
                    A = advantage + algo.gamma * A
                    sum_advantages.append(A)

                total_sums += sum_advantages
        # smaller is better
        return float(np.mean(total_sums))


class AverageValueEstimationEvaluator(EvaluatorProtocol):
    r"""Returns average value estimation.

    This metric suggests the scale for estimation of Q functions.
    If average value estimation is too large, the Q functions overestimate
    action-values, which possibly makes training failed.

    .. math::

        \mathbb{E}_{s_t \sim D} [ \max_a Q_\theta (s_t, a)]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_values = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                values = algo.predict_value(batch.observations, actions)
                total_values += values.tolist()
        return float(np.mean(total_values))


class InitialStateValueEstimationEvaluator(EvaluatorProtocol):
    r"""Returns mean estimated action-values at the initial states.

    This metric suggests how much return the trained policy would get from
    the initial states by deploying the policy to the states.
    If the estimated value is large, the trained policy is expected to get
    higher returns.

    .. math::

        \mathbb{E}_{s_0 \sim D} [Q(s_0, \pi(s_0))]

    References:
        * `Paine et al., Hyperparameter Selection for Offline Reinforcement
          Learning <https://arxiv.org/abs/2007.09055>`_

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_values = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                # estimate action-value in initial states
                first_obs = np.expand_dims(batch.observations[0], axis=0)
                actions = algo.predict(first_obs)
                values = algo.predict_value(first_obs, actions)
                total_values.append(values[0])
        return float(np.mean(total_values))


class SoftOPCEvaluator(EvaluatorProtocol):
    r"""Returns Soft Off-Policy Classification metrics.

    The metric of the scorer funciton is evaluating gaps of action-value
    estimation between the success episodes and the all episodes.
    If the learned Q-function is optimal, action-values in success episodes
    are expected to be higher than the others.
    The success episode is defined as an episode with a return above the given
    threshold.

    .. math::

        \mathbb{E}_{s, a \sim D_{success}} [Q(s, a)]
            - \mathbb{E}_{s, a \sim D} [Q(s, a)]

    References:
        * `Irpan et al., Off-Policy Evaluation via Off-Policy Classification.
          <https://arxiv.org/abs/1906.01624>`_

    Args:
        return_threshold: Return threshold of success episodes.
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _return_threshold: float
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self,
        return_threshold: float,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._return_threshold = return_threshold
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        success_values = []
        all_values = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            is_success = episode.compute_return() >= self._return_threshold
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                values = algo.predict_value(batch.observations, batch.actions)
                all_values += values.reshape(-1).tolist()
                if is_success:
                    success_values += values.reshape(-1).tolist()
        return float(np.mean(success_values) - np.mean(all_values))


class ContinuousActionDiffEvaluator(EvaluatorProtocol):
    r"""Returns squared difference of actions between algorithm and dataset.

    This metric suggests how different the greedy-policy is from the given
    episodes in continuous action-space.
    If the given episodes are near-optimal, the small action difference would
    be better.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D} [(a_t - \pi_\phi (s_t))^2]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_diffs = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                diff = ((batch.actions - actions) ** 2).sum(axis=1).tolist()
                total_diffs += diff
        return float(np.mean(total_diffs))


class DiscreteActionMatchEvaluator(EvaluatorProtocol):
    r"""Returns percentage of identical actions between algorithm and dataset.

    This metric suggests how different the greedy-policy is from the given
    episodes in discrete action-space.
    If the given episdoes are near-optimal, the large percentage would be
    better.

    .. math::

        \frac{1}{N} \sum^N \parallel
            \{a_t = \text{argmax}_a Q_\theta (s_t, a)\}

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_matches = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                match = (batch.actions.reshape(-1) == actions).tolist()
                total_matches += match
        return float(np.mean(total_matches))


class CompareContinuousActionDiffEvaluator(EvaluatorProtocol):
    r"""Action difference between algorithms.

    This metric suggests how different the two algorithms are in continuous
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D}
            [(\pi_{\phi_1}(s_t) - \pi_{\phi_2}(s_t))^2]

    Args:
        base_algo: Target algorithm to comapre with.
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _base_algo: QLearningAlgoProtocol
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self,
        base_algo: QLearningAlgoProtocol,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._base_algo = base_algo
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_diffs = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                base_actions = self._base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                diff = ((actions - base_actions) ** 2).sum(axis=1).tolist()
                total_diffs += diff
        return float(np.mean(total_diffs))


class CompareDiscreteActionMatchEvaluator(EvaluatorProtocol):
    r"""Action matches between algorithms.

    This metric suggests how different the two algorithms are in discrete
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D} [\parallel
            \{\text{argmax}_a Q_{\theta_1}(s_t, a)
            = \text{argmax}_a Q_{\theta_2}(s_t, a)\}]

    Args:
        base_algo: Target algorithm to comapre with.
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _base_algo: QLearningAlgoProtocol
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self,
        base_algo: QLearningAlgoProtocol,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._base_algo = base_algo
        self._episodes = episodes

    def __call__(
        self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
    ) -> float:
        total_matches = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                base_actions = self._base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                match = (base_actions == actions).tolist()
                total_matches += match
        return float(np.mean(total_matches))

 
class DatasetErrorEvaluator(EvaluatorProtocol):
    def __init__(self, dataset: ReplayBuffer):
        self._dataset = dataset
    
    def __call__(
        self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
    ) -> float:
        episodes = self._dataset.episodes
        res_dict ={
            'zone': [[], []],
            'own_the_ball': [[], []],
            'ball_direction': [[], []],
        }
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE * 10, self._dataset.transition_picker):
                actions = algo.predict(batch.observations)
                res_dict['zone'][0].append(np.mean(np.abs(batch.actions[:, :2] - actions[:, :2])))
                res_dict['own_the_ball'][0].append(np.mean(np.abs(batch.actions[:, 2] - actions[:, 2])))
                res_dict['zone'][1].append(np.mean(np.abs(batch.actions[:, :2] - actions[:, :2])[np.mean(np.abs(batch.actions[:, :2]), axis=-1) != 0]))
                res_dict['own_the_ball'][1].append(np.mean(np.abs(batch.actions[:, 2] - actions[:, 2])[np.abs(batch.actions[:, 2]) != 0]))
                res_dict['ball_direction'][0].append(np.mean(np.abs(batch.actions[:, 3] - actions[:, 3])[np.abs(batch.actions[:, 2]) != 0]))
                res_dict['ball_direction'][1].append(np.mean(np.abs(batch.actions[:, 3] - actions[:, 3])[np.logical_and(np.abs(batch.actions[:, 2]) != 0, np.abs(batch.actions[:, 3]) != 0)]))

        flat_res_dict = {}
        for k, v in res_dict.items():
            flat_res_dict[k] = np.nanmean(v[0])
            flat_res_dict[k + '_nzero'] = np.nanmean(v[1])
        return flat_res_dict
                

class EnsembleDatasetErrorEvaluator(EvaluatorProtocol):
    def __init__(self, dataset: ReplayBuffer):
        self._dataset = dataset
    
    def __call__(
        self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
    ) -> float:
        res_dict ={
            'obs_mse': [],
            'reward_mse': [],
            'obs_uncertainty': [],
            'reward_uncertainty': [],

        }
        for i in range(10):
            samples = self._dataset.sample_transition_batch(10240)
            pred_obs_rew = algo.predict(samples.observations)
            pred_obs_rew_uncertainty = algo.predict_uncertainty(samples.observations)
            obs_mse = (pred_obs_rew[:, :-1] - samples.actions[:, :-1])**2
            reward_mse = (pred_obs_rew[:, -1] - samples.actions[:, -1])**2
            obs_uncertainty = pred_obs_rew_uncertainty[:, :-1]
            reward_uncertainty = pred_obs_rew_uncertainty[:, -1]
            res_dict['obs_mse'].append(obs_mse)
            res_dict['reward_mse'].append(reward_mse)
            res_dict['obs_uncertainty'].append(obs_uncertainty)
            res_dict['reward_uncertainty'].append(reward_uncertainty)
        flat_res_dict = {}
        for k, v in res_dict.items():
            flat_res_dict[k + '_mean'] = np.nanmean(v)
            flat_res_dict[k + '_max'] = np.max(v)
            flat_res_dict[k + '_min'] = np.min(v)
            flat_res_dict[k + '_std'] = np.std(v)
        return flat_res_dict
                
        
    
class EnvironmentEvaluator(EvaluatorProtocol):
    r"""Action matches between algorithms.

    This metric suggests how different the two algorithms are in discrete
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D} [\parallel
            \{\text{argmax}_a Q_{\theta_1}(s_t, a)
            = \text{argmax}_a Q_{\theta_2}(s_t, a)\}]

    Args:
        env: Gym environment.
        n_trials: Number of episodes to evaluate.
        epsilon: Probability of random action.
    """
    _env: GymEnv
    _n_trials: int
    _epsilon: float

    def __init__(
        self,
        env: GymEnv,
        n_trials: int = 10,
        epsilon: float = 0.0,
        obs_type: str = 'raw',
        update_stack_obs=None,
        stack_obs_len=None,
        acs_replace_strategy='parts_keeped',
        
        llm=False,
        llm_version: str = 'baseline',
        rule_based=False,
    ):
        self._env = env
        self._n_trials = n_trials
        self._epsilon = epsilon
        self._obs_type = obs_type
        self.update_stack_obs = update_stack_obs
        self.stack_obs_len = stack_obs_len
        self.acs_replace_strategy = acs_replace_strategy
        
        self.llm = llm
        self.llm_version = llm_version
        self.rule_based = rule_based

    def __call__(
        self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
    ) -> float:
        if isinstance(algo, QLearningAlgoProtocol):
            if self._obs_type == 'raw':
                return evaluate_qlearning_with_environment(
                    algo=algo,
                    env_list=self._env,
                    n_trials=self._n_trials,
                    epsilon=self._epsilon,
                )
            elif self._obs_type == 'imginary_obs':
                return evaluate_qlearning_with_img_obs_environment(
                    algo=algo,
                    env_list=self._env,
                    n_trials=self._n_trials,
                    epsilon=self._epsilon,
                    update_stack_obs=self.update_stack_obs,
                    stack_obs_len=self.stack_obs_len,
                    acs_replace_strategy=self.acs_replace_strategy,
                )
        elif self.rule_based:
            return evaluate_rule_based_with_environment(
                env_list=self._env,
                n_trials=self._n_trials,
            )
        elif self.llm:
            return evaluate_llm_with_environment(
                env_list=self._env,
                n_trials=self._n_trials,
                llm_version=self.llm_version,
            )
            

class TTTEnvironmentEvaluator(EvaluatorProtocol):
    r"""Action matches between algorithms.

    This metric suggests how different the two algorithms are in discrete
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D} [\parallel
            \{\text{argmax}_a Q_{\theta_1}(s_t, a)
            = \text{argmax}_a Q_{\theta_2}(s_t, a)\}]

    Args:
        env: Gym environment.
        n_trials: Number of episodes to evaluate.
        epsilon: Probability of random action.
    """
    _env: GymEnv
    _n_trials: int
    _epsilon: float

    def __init__(
        self,
        env: GymEnv,
        oppo_players: list,
        n_trials: int = 10,
        epsilon: float = 0.0,
    ):
        self._env = env
        self._n_trials = n_trials
        self._epsilon = epsilon
        self.oppo_players = oppo_players

    def __call__(
        self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
    ) -> float:
        return evaluate_ttt_environment(
            algo=algo,
            env=self._env,
            oppo_players=self.oppo_players,
            n_trials=self._n_trials,
            epsilon=self._epsilon,
        )