import dataclasses
from typing import Dict
from abc import abstractmethod
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

import ipdb
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm, trange
from typing_extensions import Self

import d3rlpy
from d3rlpy.base import LearnableConfig, save_config
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import (
    DatasetInfo,
    ReplayBuffer,
)
from d3rlpy.logging import (
    LOG,
    D3RLPyLogger,
    FileAdapterFactory,
    LoggerAdapterFactory,
)
from d3rlpy.metrics import EvaluatorProtocol, evaluate_qlearning_with_environment
from d3rlpy.models.torch import EnsembleQFunction, Policy
from d3rlpy.torch_utility import (
    TorchMiniBatch,
)
from d3rlpy.algos.utility import (
    assert_action_space_with_dataset,
    build_scalers_with_transition_picker,
)

from d3rlpy.base import DeviceArg, LearnableConfig, register_learnable
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from d3rlpy.dataset import Shape
from d3rlpy.models.builders import create_discrete_q_function
from d3rlpy.models.encoders import EncoderFactory, make_encoder_field
from d3rlpy.models.optimizers import OptimizerFactory, make_optimizer_field
from d3rlpy.models.q_functions import QFunctionFactory, make_q_func_field
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.dataset.transition_pickers import BasicTransitionPicker

from offline_active_rl.active_cql_impl import DiscreteActiveCQLImpl
import offline_active_rl.acquisitions as acquisitions

__all__ = ["DiscreteActiveCQLConfig", "DiscreteActiveCQL"]


@dataclasses.dataclass()
class DiscreteActiveCQLConfig(LearnableConfig):
    r"""Config of Discrete Active version of Conservative Q-Learning algorithm.

    Discrete Active version of CQL is a DoubleDQN-based data-driven deep reinforcement
    learning algorithm (the original paper uses DQN), which achieves
    state-of-the-art performance in offline RL problems. The Sampling of data points
    is guided by the Q-function uncertainty estimates. 

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::

        L(\theta) = \alpha \mathbb{E}_{s_t \sim D}
            [\log{\sum_a \exp{Q_{\theta}(s_t, a)}}
             - \mathbb{E}_{a \sim D} [Q_{\theta}(s, a)]]
            + L_{DoubleDQN}(\theta)

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        learning_rate (float): Learning rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        n_critics (int): Number of Q functions for ensemble.
        target_update_interval (int): Interval to synchronize the target
            network.
        alpha (float): math:`\alpha` value above.
        num_bkwds (int): Number of backward steps.
        acq_func (str): Name of acquisition function. The candidates are
            ``'random'``, ``'mean'``, ``'bald'``, ``'max'``, ``'variance'``
        init_rand_sample_epochs (int): Number of epochs to train on randomly
            sampled data at the beginning of training.
        indep_ensemble (bool): Flag to use independent ensemble. If ``True``,
            each Q-function is trained with its own targets.
    """
    learning_rate: float = 6.25e-5
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 32
    gamma: float = 0.99
    n_critics: int = 1
    target_update_interval: int = 8000
    alpha: float = 1.0
    num_bkwds: int = 1
    acq_func: str = "random"
    init_rand_sample_epochs: int = 0
    indep_ensemble: bool = False

    def create(self, device: DeviceArg = False) -> "DiscreteActiveCQL":
        return DiscreteActiveCQL(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_active_cql"


class DiscreteActiveCQL(QLearningAlgoBase[DiscreteActiveCQLImpl, DiscreteActiveCQLConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        q_func = create_discrete_q_function(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )

        optim = self._config.optim_factory.create(
            q_func.parameters(), lr=self._config.learning_rate
        )

        self._impl = DiscreteActiveCQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            q_func=q_func,
            optim=optim,
            gamma=self._config.gamma,
            alpha=self._config.alpha,
            device=self._device,
        )
    
    def fitter(
        self,
        dataset: ReplayBuffer,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
        iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            Iterator yielding current epoch and metrics dict.
        """
        dataset_info = DatasetInfo.from_episodes(dataset.episodes)
        LOG.info("dataset info", dataset_info=dataset_info)

        # check action space
        assert_action_space_with_dataset(self, dataset_info)

        # initialize scalers
        build_scalers_with_transition_picker(self, dataset)

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__
        logger = D3RLPyLogger(
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            with_timestamp=with_timestamp,
        )

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = dataset_info.action_size
            observation_shape = (
                dataset.sample_transition().observation_signature.shape[0]
            )
            self.create_impl(observation_shape, action_size)
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        save_config(self, logger)

        # active sampling related params ################################################
        num_bkwds = self._config.num_bkwds
        fwd_batch_size = self._config.batch_size
        batch_size = self._config.batch_size
        dataset_size = dataset.transition_count     
        n_batch = dataset_size // fwd_batch_size + 1
        total_idxs = list(range(dataset_size))
        transition_picker = BasicTransitionPicker()

        all_actions_list = []
        all_obs_list = []
        eps_transition_indices_list = []

        # loop over dataset episodes and concatenate actions
        for i in range(len(dataset.episodes)):
            actions = torch.tensor(dataset.episodes[i].actions, dtype=torch.int64).to('cuda')
            all_actions_list.append(actions)
            obs = torch.tensor(dataset.episodes[i].observations, dtype=torch.float32).to('cuda')
            all_obs_list.append(obs)

            for j in range(dataset.episodes[i].transition_count):
                eps_transition_indices_list.append((i, j))
        
        all_actions = torch.cat(all_actions_list)
        all_obs = torch.cat(all_obs_list)

        ###############################################################################



        # training loop
        n_epochs = n_steps // n_steps_per_epoch
        total_step = 0
        for epoch in range(1, n_epochs + 1):
            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            if epoch <= self._config.init_rand_sample_epochs:
                aq_func = 'random'
            else:
                aq_func = self._config.acq_func

            range_gen = tqdm(
                range(n_steps_per_epoch),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            for itr in range_gen:
                with logger.measure_time("step"):
                    with torch.no_grad():
                        if aq_func=='random':  
                            mu = torch.zeros(len(self._impl.q_function.q_funcs), dataset_size, dataset_info.action_size).to('cuda')
                        else:
                            outputs = []
                            for j in range(n_batch):
                                batch_idxs = total_idxs[j * fwd_batch_size : (j + 1) * fwd_batch_size]
                                mu = self._impl._q_func(
                                    self.observation_scaler.transform(all_obs[batch_idxs]) if self.observation_scaler else obs,       #####
                                    reduction='none')
                                                                    
                                outputs.append(mu)
                            mu = torch.cat(outputs, dim=1)

                        scores = acquisitions.FUNCTIONS[aq_func](mu, all_actions)
                        scores = scores + scores.min().abs()

                        scores = scores.cpu().numpy()
                        p = (scores / (scores.sum()*1.0)).astype('float64')

                    for i_bkwd in range(num_bkwds):
                        idx_select = np.random.choice(
                            range(len(p)),
                            replace=False,
                            p=p,
                            size=batch_size,
                        )

                        batch_start_transitions = []
                        for each_chosen_idx in idx_select:
                            episode_index, transition_index = eps_transition_indices_list[each_chosen_idx]
                            batch_start_transitions.append(transition_picker(dataset.episodes[episode_index], transition_index))     
                        selected_batch = TransitionMiniBatch.from_transitions(batch_start_transitions)

                        # validate all this 

                        # # pick transitions
                        # with logger.measure_time("sample_batch"):
                        #     batch = dataset.sample_transition_batch(
                        #         self._config.batch_size
                        #     )

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            loss = self.update(selected_batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # call epoch_callback if given
            if epoch_callback:
                epoch_callback(self, epoch, total_step)

            if evaluators:
                for name, evaluator in evaluators.items():
                    test_score = evaluator(self, dataset)
                    if name=='environment':
                        total_reward = []
                        rew_dict = test_score
                        for key, value in test_score.items():
                            logger.add_metric(key, value)
                            total_reward.append(value)
                        logger.add_metric('environment', np.mean(total_reward))
                    else:
                        # logging metrics
                        logger.add_metric(name, test_score)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            log_dict = {
                "loss": metrics["loss"],
                "reward": metrics["environment"], 
                "conservative_loss": metrics["conservative_loss"],    
            }

            for key, value in rew_dict.items():
                log_dict['reward_' + key] = value
            
            wandb.log(
                log_dict,
                step=epoch*n_steps_per_epoch,
            )

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        logger.close()

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        loss, conservative_loss = self._impl.update(batch)
        if self._grad_step % self._config.target_update_interval == 0:
            self._impl.update_target()
        return {"loss": loss, "conservative_loss": conservative_loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(DiscreteActiveCQLConfig)