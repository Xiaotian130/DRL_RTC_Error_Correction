# -*- coding: utf-8 -*-
"""
Training script for the error-compensation policy used in the SWMM-based
reinforcement learning framework.

This script trains a compensation policy that adjusts baseline control
actions under perturbed observations. The training reward is defined
based on the flooding reduction relative to a baseline rollout under
the same observation error.

Note
----
Some lower-level reinforcement-learning components used in this workflow
are adapted from previously released GPL-licensed implementations.
Please see LICENSE and NOTICE for attribution details.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

import swmm_env
import buffer
import ppo
import ppo_error


# ============================================================
# TensorFlow setup
# ============================================================
tf.compat.v1.reset_default_graph()
tf.compat.v1.enable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)


# ============================================================
# Configuration
# ============================================================
ENV_PARAMS = {
    "orf": "tiaoxu1",
    "parm": "./states_yaml/tiaoxu1",
    "GI": False,
    "advance_seconds": 300,
    "kf": 1,
    "kc": 1,
    "train": False,
    "reward_epsilon": 1e-8,
    "base_dir": ".",
    "swmm_dir": "./SWMM",
    "temp_train_dir": "./_teminp",
    "temp_test_dir": "./_temtestinp",
}

TRAIN_CONFIG = {
    "rainfall_path": "./training_rainfall/training_raindata_624.npy",
    "baseline_model_dir": "./models/baseline",
    "error_model_dir": "./models/error_model_train",
    "results_dir": "./results/",
    "train_run_name": "error_compensation_run",
    "initialize_error_model": False,
    "n_jobs": 1,
}

OBSERVATION_ERROR_LEVELS = [-0.1]


# ============================================================
# Configuration builders
# ============================================================
def build_agent_configs(env):
    """
    Build configuration dictionaries for the baseline controller and
    the error-compensation controller.
    """
    state_dim = len(env.config["states"])

    baseline_agent_config = {
        "state_dim": state_dim,
        "action_dim": 1331,
        "actornet_layer": [50, 50, 50],
        "criticnet_layer": [50, 50, 50],
        "bound_low": 0,
        "bound_high": 2,
        "clip_ratio": 0.01,
        "target_kl": 0.1,
        "lam": 0.9,
        "policy_learning_rate": 0.00005,
        "value_learning_rate": 0.00005,
        "train_policy_iterations": 10,
        "train_value_iterations": 10,
        "num_rain": 1,
        "training_step": 1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "ep_min": 1e-5,
        "ep_decay": 0.99,
    }

    compensation_agent_config = {
        "state_dim": state_dim,
        "action_dim": 9261,
        "actornet_layer": [50, 50, 50],
        "criticnet_layer": [50, 50, 50],
        "bound_low": 0,
        "bound_high": 2,
        "clip_ratio": 0.01,
        "target_kl": 0.1,
        "lam": 0.9,
        "policy_learning_rate": 0.001,
        "value_learning_rate": 0.001,
        "train_policy_iterations": 10,
        "train_value_iterations": 10,
        "num_rain": 1,
        "num_error1": 1,
        "training_step": 1,
        "gamma": 0.9,
        "epsilon": 0.1,
        "ep_min": 1e-100,
        "ep_decay": 0.99,
    }

    return baseline_agent_config, compensation_agent_config


# ============================================================
# Utility functions
# ============================================================
def ensure_dir(path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def apply_observation_error(observation, error_ratio):
    """
    Apply multiplicative observation error to the state vector.

    In the original implementation, the final element of the observation
    vector is excluded from perturbation.
    """
    perturbed_observation = list(observation)
    for idx in range(len(perturbed_observation) - 1):
        perturbed_observation[idx] *= (1 + error_ratio)
    return perturbed_observation


def load_baseline_model(agent_config, model_dir):
    """Load the pretrained baseline controller."""
    model = ppo.PPO(agent_config)
    model.load_model(model_dir)
    return model


def initialize_compensation_model(agent_config, model_dir, initialize_weights=False):
    """
    Initialize the compensation model and optionally create weight files
    before loading the model state.
    """
    model = ppo_error.PPO(agent_config)

    if initialize_weights:
        model.save_model(model_dir)

    model.load_model(model_dir)
    return model


# ============================================================
# Rollout functions
# ============================================================
def run_baseline_rollout(env, baseline_model, rainfall_event, rainfall_index, observation_error):
    """
    Run a baseline simulation under observation error only.

    Returns
    -------
    list
        Flooding time series produced by the baseline controller.
    """
    flooding_series = []

    observation = env.reset(rainfall_event, rainfall_index, True, "train")
    done = False

    while not done:
        perturbed_observation = apply_observation_error(observation, observation_error)
        perturbed_observation = np.array(perturbed_observation).reshape(1, -1)

        _, baseline_action_index = ppo.sample_action(
            perturbed_observation, baseline_model, False
        )
        baseline_action = baseline_model.action_table[
            int(baseline_action_index[0].numpy())
        ].tolist()

        observation, _, flooding_value, _, _, _, done = env.step(baseline_action)
        flooding_series.append(flooding_value)

    return flooding_series


def combine_baseline_and_compensation_actions(baseline_action, compensation_action):
    """
    Combine baseline control action and compensation action.

    Note
    ----
    This function preserves the original multiplicative logic used in the
    study code. If a one-to-one action correction scheme is intended,
    this function should be revised accordingly.
    """
    corrected_action = baseline_action.copy()

    for action_idx in range(len(corrected_action)):
        for correction_idx in range(len(compensation_action)):
            corrected_action[action_idx] *= (1 + compensation_action[correction_idx])

    return corrected_action


def run_compensation_rollout(
    env,
    baseline_model,
    compensation_model,
    rainfall_event,
    rainfall_index,
    observation_error,
):
    """
    Run the two-stage rollout used for training the compensation policy.

    Stage 1:
        Run the baseline controller under observation error only and
        record the flooding time series.

    Stage 2:
        Run the baseline controller again under the same observation
        error, but modify the baseline action using the compensation
        policy. Reward is computed from flooding reduction.

    Returns
    -------
    tuple
        States, actions, rewards, values, log probabilities, last value,
        episode return, episode length, baseline flooding series,
        corrected flooding series.
    """
    states, actions, rewards = [], [], []
    values, log_probs = [], []
    corrected_flooding_series = []

    baseline_flooding_series = run_baseline_rollout(
        env=env,
        baseline_model=baseline_model,
        rainfall_event=rainfall_event,
        rainfall_index=rainfall_index,
        observation_error=observation_error,
    )

    observation = env.reset(rainfall_event, rainfall_index, True, "train")
    done = False
    step_index = 0
    episode_return = 0.0
    episode_length = 0

    while not done:
        perturbed_observation = apply_observation_error(observation, observation_error)
        perturbed_observation = np.array(perturbed_observation).reshape(1, -1)

        _, baseline_action_index = ppo.sample_action(
            perturbed_observation, baseline_model, False
        )
        compensation_logits, compensation_action_index = ppo_error.sample_action(
            perturbed_observation, compensation_model, True
        )

        baseline_action = baseline_model.action_table[
            int(baseline_action_index[0].numpy())
        ].tolist()

        compensation_action = compensation_model.action_table[
            int(compensation_action_index[0].numpy())
        ].tolist()

        corrected_action = combine_baseline_and_compensation_actions(
            baseline_action, compensation_action
        )

        observation, _, corrected_flooding, _, _, _, done = env.step(corrected_action)

        baseline_flooding = baseline_flooding_series[step_index]
        baseline_max = max(baseline_flooding_series) if baseline_flooding_series else 1.0
        baseline_max = baseline_max if baseline_max != 0 else 1.0

        reward = (baseline_flooding - corrected_flooding) / baseline_max

        value = compensation_model.critic(perturbed_observation)
        log_prob = ppo_error.logprobabilities(
            compensation_logits,
            compensation_action_index,
            compensation_model.params["action_dim"],
        )

        states.append(perturbed_observation)
        actions.append(compensation_action_index.numpy().tolist())
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        corrected_flooding_series.append(corrected_flooding)

        episode_return += reward
        episode_length += 1
        step_index += 1

    last_value = 0 if done else compensation_model.critic(observation.reshape(1, -1))

    return (
        states,
        actions,
        rewards,
        values,
        log_probs,
        last_value,
        episode_return,
        episode_length,
        baseline_flooding_series,
        corrected_flooding_series,
    )


def interact_episode(
    rainfall_index,
    error_index,
    epsilon,
    observation_error_levels,
    env_params,
    baseline_agent_config,
    compensation_agent_config,
    train_config,
    rainfall_data,
):
    """
    Execute one interaction episode for a given rainfall event and
    observation error level.
    """
    baseline_model = load_baseline_model(
        baseline_agent_config,
        train_config["baseline_model_dir"],
    )

    compensation_model = ppo_error.PPO(compensation_agent_config)
    compensation_model.load_model(train_config["error_model_dir"])
    compensation_model.params["epsilon"] = epsilon

    env = swmm_env.SWMM_ENV(env_params)
    rainfall_event = rainfall_data[rainfall_index]
    observation_error = observation_error_levels[error_index]

    return run_compensation_rollout(
        env=env,
        baseline_model=baseline_model,
        compensation_model=compensation_model,
        rainfall_event=rainfall_event,
        rainfall_index=rainfall_index,
        observation_error=observation_error,
    )


# ============================================================
# Training procedure
# ============================================================
def train():
    """Main training entry point."""
    ensure_dir(TRAIN_CONFIG["results_dir"])
    ensure_dir(TRAIN_CONFIG["error_model_dir"])

    env = swmm_env.SWMM_ENV(ENV_PARAMS)
    rainfall_data = np.load(
        TRAIN_CONFIG["rainfall_path"],
        allow_pickle=True
    ).tolist()

    baseline_agent_config, compensation_agent_config = build_agent_configs(env)

    compensation_model = initialize_compensation_model(
        agent_config=compensation_agent_config,
        model_dir=TRAIN_CONFIG["error_model_dir"],
        initialize_weights=TRAIN_CONFIG["initialize_error_model"],
    )
    print("Compensation model loaded.")

    history = {
        "episode": [],
        "Episode_reward": [],
        "Episode_flooding": [],
    }

    for epoch in range(compensation_model.params["training_step"]):
        total_return = 0.0
        total_steps = 0
        total_episodes = 0
        last_flooding = 0.0

        trajectory_buffer = buffer.Buffer(
            compensation_model.params["state_dim"],
            int(
                len(rainfall_data[0])
                * compensation_model.params["num_rain"]
                * compensation_model.params["num_error1"]
            ),
        )

        parameter_grid = [
            (rainfall_index, error_index)
            for rainfall_index in range(compensation_model.params["num_rain"])
            for error_index in range(compensation_model.params["num_error1"])
        ]

        results = Parallel(n_jobs=TRAIN_CONFIG["n_jobs"])(
            delayed(interact_episode)(
                rainfall_index=rainfall_index,
                error_index=error_index,
                epsilon=compensation_model.params["epsilon"],
                observation_error_levels=OBSERVATION_ERROR_LEVELS,
                env_params=ENV_PARAMS,
                baseline_agent_config=baseline_agent_config,
                compensation_agent_config=compensation_agent_config,
                train_config=TRAIN_CONFIG,
                rainfall_data=rainfall_data,
            )
            for rainfall_index, error_index in parameter_grid
        )

        for result in results:
            (
                states,
                actions,
                rewards,
                values,
                log_probs,
                last_value,
                episode_return,
                episode_length,
                baseline_flooding_series,
                corrected_flooding_series,
            ) = result

            for state, action, reward, value, log_prob in zip(
                states, actions, rewards, values, log_probs
            ):
                trajectory_buffer.store(state, action, reward, value, log_prob)

            trajectory_buffer.finish_trajectory(last_value)
            total_return += episode_return
            total_steps += episode_length
            total_episodes += 1

            if baseline_flooding_series:
                last_flooding = baseline_flooding_series[-1]

        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = trajectory_buffer.get()

        for _ in range(compensation_model.params["train_policy_iterations"]):
            ppo_error.train_policy(
                observation_buffer,
                action_buffer,
                logprobability_buffer,
                advantage_buffer,
                compensation_model,
            )

        for _ in range(compensation_model.params["train_value_iterations"]):
            ppo_error.train_value_function(
                observation_buffer,
                return_buffer,
                compensation_model,
            )

        compensation_model.save_model(TRAIN_CONFIG["error_model_dir"])

        history["episode"].append(epoch)
        history["Episode_reward"].append(total_return)
        history["Episode_flooding"].append(last_flooding)

        if (
            compensation_model.params["epsilon"] >= compensation_model.params["ep_min"]
            and epoch % 3 == 0
        ):
            compensation_model.params["epsilon"] *= compensation_model.params["ep_decay"]

        print(
            f"Epoch: {epoch + 1}, "
            f"Return: {total_return:.6f}, "
            f"Flooding: {last_flooding:.6f}, "
            f"Episodes: {total_episodes}, "
            f"Steps: {total_steps}"
        )

        history_path = (
            Path(TRAIN_CONFIG["results_dir"])
            / f"train_history_{TRAIN_CONFIG['train_run_name']}.npy"
        )
        np.save(history_path, history)

    figure_path = (
        Path(TRAIN_CONFIG["results_dir"])
        / f"train_reward_{TRAIN_CONFIG['train_run_name']}.png"
    )

    plt.figure()
    plt.plot(history["Episode_reward"])
    plt.xlabel("Epoch")
    plt.ylabel("Episode reward")
    plt.title("Training reward history")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train()