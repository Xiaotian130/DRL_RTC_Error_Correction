# -*- coding: utf-8 -*-
"""
Testing script for the SWMM-based error-compensation control framework.

This script evaluates the compensation framework by comparing:
1. baseline control under perturbed observations, and
2. compensated control under the same observation error.

A compensation model is selected according to the predefined observation
error interval.

Note
----
Some lower-level reinforcement-learning components used in this workflow
are adapted from previously released GPL-licensed implementations.
Please see LICENSE and NOTICE for attribution details.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import swmm_env
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

TEST_CONFIG = {
    "rainfall_path": "./examples/test_raindata.npy",
    "baseline_model_dir": "./models/baseline",
    "results_dir": "./results/test",
    "test_id": "example_test",
    "num_tests": 1,
    "rainfall_index": 1,
}

OBSERVATION_ERROR_CANDIDATES = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]


# ============================================================
# Configuration builders
# ============================================================
def build_agent_configs(env):
    """
    Build configuration dictionaries for the baseline controller and
    the compensation controller.
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
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def apply_observation_error(observation, error_ratio):
    """
    Apply multiplicative observation error to the observation vector.

    The final element is not perturbed, following the original code.
    """
    perturbed_observation = list(observation)
    for idx in range(len(perturbed_observation) - 1):
        perturbed_observation[idx] *= (1 + error_ratio)
    return perturbed_observation


def combine_baseline_and_compensation_actions(baseline_action, compensation_action):
    """
    Combine baseline action and compensation action using the original
    multiplicative logic from the study code.
    """
    corrected_action = baseline_action.copy()

    for action_idx in range(len(corrected_action)):
        for correction_idx in range(len(compensation_action)):
            corrected_action[action_idx] *= (1 + compensation_action[correction_idx])

    return corrected_action


def load_baseline_model(agent_config, model_dir):
    """Load the pretrained baseline controller."""
    model = ppo.PPO(agent_config)
    model.load_model(model_dir)
    return model


def load_compensation_model(agent_config, model_dir):
    """Load one compensation model."""
    model = ppo_error.PPO(agent_config)
    model.load_model(model_dir)
    return model


def build_compensation_model_map(agent_config):
    """
    Build a mapping between observation-error intervals and
    corresponding compensation model directories.
    """
    return [
        ((-0.15, 0.0), load_compensation_model(agent_config, "./models/error_model_1")),
        ((-0.25, -0.15), load_compensation_model(agent_config, "./models/error_model_2")),
        ((-0.35, -0.25), load_compensation_model(agent_config, "./models/error_model_3")),
        ((-0.45, -0.35), load_compensation_model(agent_config, "./models/error_model_4")),
        ((-0.55, -0.45), load_compensation_model(agent_config, "./models/error_model_5")),
        ((-0.65, -0.55), load_compensation_model(agent_config, "./models/error_model_6")),
        ((-0.75, -0.65), load_compensation_model(agent_config, "./models/error_model_7")),
        ((-0.85, -0.75), load_compensation_model(agent_config, "./models/error_model_8")),
        ((-1.00, -0.85), load_compensation_model(agent_config, "./models/error_model_9")),
    ]


def select_compensation_model(error_value, model_map):
    """
    Select the compensation model corresponding to the given
    observation error.
    """
    for (lower_bound, upper_bound), model in model_map:
        if lower_bound <= error_value < upper_bound:
            return model

    raise ValueError(f"No compensation model found for error value: {error_value}")


# ============================================================
# Rollout functions
# ============================================================
def run_baseline_test_rollout(env, baseline_model, rainfall_event, rainfall_index, test_id, observation_error):
    """
    Run the baseline controller under observation error only.
    """
    history = {
        "time": [],
        "state": [],
        "action": [],
        "reward": [],
        "flooding": [],
        "main_flow": [],
        "main_flooding": [],
    }

    observation = env.reset(rainfall_event, rainfall_index, False, test_id)
    done = False
    time_step = 0

    history["time"].append(time_step)
    history["state"].append(observation)

    while not done:
        perturbed_observation = apply_observation_error(observation, observation_error)
        perturbed_observation = np.array(perturbed_observation).reshape(1, -1)

        _, action_index = ppo.sample_action(perturbed_observation, baseline_model, False)
        baseline_action = baseline_model.action_table[int(action_index[0].numpy())].tolist()

        for _ in range(int(300 / env.params["advance_seconds"])):
            (
                observation_new,
                reward,
                flooding_value,
                main_flow,
                main_flooding,
                storageing_volume,
                flooding_volume,
                done,
            ) = env.twostep(baseline_action)

            if done:
                break

            time_step += 1
            history["time"].append(time_step)
            history["state"].append(observation)
            history["action"].append(baseline_action)
            history["reward"].append(reward)
            history["main_flow"].append(main_flow)
            history["main_flooding"].append(main_flooding)
            history["flooding"].append(flooding_value)

        observation = observation_new

    return history


def run_compensated_test_rollout(
    env,
    baseline_model,
    compensation_model,
    rainfall_event,
    rainfall_index,
    test_id,
    observation_error,
):
    """
    Run the compensated controller under the same observation error.
    """
    history = {
        "time": [],
        "state": [],
        "baseline_action": [],
        "corrected_action": [],
        "reward": [],
        "flooding": [],
        "main_flow": [],
        "main_flooding": [],
        "compensation_action": [],
    }

    observation = env.reset(rainfall_event, rainfall_index, False, test_id)
    done = False
    time_step = 0

    history["time"].append(time_step)
    history["state"].append(observation)

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

        for _ in range(int(300 / env.params["advance_seconds"])):
            (
                observation_new,
                reward,
                flooding_value,
                main_flow,
                main_flooding,
                storageing_volume,
                flooding_volume,
                done,
            ) = env.twostep(corrected_action)

            if done:
                break

            time_step += 1
            history["time"].append(time_step)
            history["state"].append(observation)
            history["baseline_action"].append(baseline_action)
            history["corrected_action"].append(corrected_action)
            history["reward"].append(reward)
            history["main_flow"].append(main_flow)
            history["main_flooding"].append(main_flooding)
            history["flooding"].append(flooding_value)
            history["compensation_action"].append(compensation_action)

        observation = observation_new

    return history


# ============================================================
# Evaluation function
# ============================================================
def evaluate_one_case(
    rainfall_event,
    rainfall_index,
    test_id,
    observation_error,
    env_params,
    baseline_model,
    compensation_model_map,
):
    """
    Evaluate one rainfall case under one observation-error level.
    """
    compensation_model = select_compensation_model(
        observation_error,
        compensation_model_map
    )

    baseline_env = swmm_env.SWMM_ENV(env_params)
    compensated_env = swmm_env.SWMM_ENV(env_params)

    baseline_history = run_baseline_test_rollout(
        env=baseline_env,
        baseline_model=baseline_model,
        rainfall_event=rainfall_event,
        rainfall_index=rainfall_index,
        test_id=test_id,
        observation_error=observation_error,
    )

    compensated_history = run_compensated_test_rollout(
        env=compensated_env,
        baseline_model=baseline_model,
        compensation_model=compensation_model,
        rainfall_event=rainfall_event,
        rainfall_index=rainfall_index,
        test_id=test_id,
        observation_error=observation_error,
    )

    max_baseline_flooding = max(baseline_history["flooding"]) if baseline_history["flooding"] else 0.0
    max_compensated_flooding = max(compensated_history["flooding"]) if compensated_history["flooding"] else 0.0

    return {
        "observation_error": observation_error,
        "max_baseline_flooding": max_baseline_flooding,
        "max_compensated_flooding": max_compensated_flooding,
        "baseline_actions": baseline_history["action"],
        "compensated_actions": compensated_history["corrected_action"],
        "baseline_history": baseline_history,
        "compensated_history": compensated_history,
    }


# ============================================================
# Main testing entry point
# ============================================================
def main():
    """Main testing entry point."""
    ensure_dir(TEST_CONFIG["results_dir"])

    env = swmm_env.SWMM_ENV(ENV_PARAMS)
    rainfall_data = np.load(TEST_CONFIG["rainfall_path"], allow_pickle=True).tolist()

    baseline_agent_config, compensation_agent_config = build_agent_configs(env)

    baseline_model = load_baseline_model(
        baseline_agent_config,
        TEST_CONFIG["baseline_model_dir"],
    )

    compensation_model_map = build_compensation_model_map(compensation_agent_config)

    results = []

    for test_index in range(TEST_CONFIG["num_tests"]):
        observation_error = np.random.choice(OBSERVATION_ERROR_CANDIDATES)

        result = evaluate_one_case(
            rainfall_event=rainfall_data[TEST_CONFIG["rainfall_index"]],
            rainfall_index=test_index,
            test_id=TEST_CONFIG["test_id"],
            observation_error=observation_error,
            env_params=ENV_PARAMS,
            baseline_model=baseline_model,
            compensation_model_map=compensation_model_map,
        )
        results.append(result)

    output_rows = []
    for result in results:
        output_rows.append(
            {
                "observation_error": result["observation_error"],
                "baseline_peak_flooding": result["max_baseline_flooding"],
                "compensated_peak_flooding": result["max_compensated_flooding"],
                "baseline_actions": result["baseline_actions"],
                "compensated_actions": result["compensated_actions"],
            }
        )

    output_df = pd.DataFrame(output_rows)
    output_file = Path(TEST_CONFIG["results_dir"]) / "test_summary.csv"
    output_df.to_csv(output_file, index=False)

    print(f"Testing finished. Results saved to: {output_file}")


if __name__ == "__main__":
    main()