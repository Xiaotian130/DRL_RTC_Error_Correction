# -*- coding: utf-8 -*-
"""
This file is adapted from previously released open-source research code.

Original source repository/project:
DRL_state_selection_cost
https://github.com/DantEzio/DRL_state_selection_cost

Original license:
GNU General Public License v3.0 (GPL v3)

Modifications made in the present study:
- used as the compensation-policy module in the SWMM-based
  observation-error control framework,
- configured to load a compensation action table,
- used together with the study-specific training and testing workflows
  for error-compensation learning.

This redistributed derivative file is provided under the GPL v3 in
accordance with the license of the original source.

Please also see LICENSE and NOTICE for attribution and provenance details.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def build_mlp(x, sizes, activation=tf.tanh, output_activation=None):
    """
    Build a feedforward multilayer perceptron.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    sizes : list[int]
        Layer sizes, including the output layer.
    activation : callable
        Activation function for hidden layers.
    output_activation : callable or None
        Activation function for the output layer.

    Returns
    -------
    tf.Tensor
        Output tensor.
    """
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, actions, num_actions):
    """
    Compute log-probabilities of selected discrete actions.

    Parameters
    ----------
    logits : tf.Tensor
        Policy logits with shape [batch_size, num_actions].
    actions : tf.Tensor
        Selected action indices.
    num_actions : int
        Number of discrete actions.

    Returns
    -------
    tf.Tensor
        Log-probabilities of the selected actions.
    """
    assert logits.shape[-1] == num_actions, (
        f"Expected logits shape to have last dimension {num_actions}, "
        f"but got {logits.shape[-1]}"
    )

    logprob_all = tf.nn.log_softmax(logits)
    logprob = tf.reduce_sum(tf.one_hot(actions, num_actions) * logprob_all, axis=1)
    return logprob


class PPO:
    """
    PPO agent used as the error-compensation policy.

    The agent outputs discrete compensation actions, which are mapped
    to action-correction factors through `error_action_table.csv`.
    """

    def __init__(self, params, action_table_path="./error_action_table.csv"):
        self.params = params
        self.action_table = pd.read_csv(action_table_path).values[:, 1:]

        self.policy_optimizer = Adam(
            learning_rate=self.params["policy_learning_rate"]
        )
        self.value_optimizer = Adam(
            learning_rate=self.params["value_learning_rate"]
        )

        self.observation_dim = self.params["state_dim"]
        self.num_actions = self.params["action_dim"]

        observation_input = keras.Input(
            shape=(self.observation_dim,),
            dtype=tf.float32,
            name="observation_input"
        )

        logits = build_mlp(
            observation_input,
            self.params["actornet_layer"] + [self.num_actions],
            activation=tf.tanh,
            output_activation=None,
        )
        self.actor = keras.Model(inputs=observation_input, outputs=logits)

        value = tf.squeeze(
            build_mlp(
                observation_input,
                self.params["criticnet_layer"] + [1],
                activation=tf.tanh,
                output_activation=None,
            ),
            axis=1,
        )
        self.critic = keras.Model(inputs=observation_input, outputs=value)

    def load_model(self, model_dir):
        """
        Load actor and critic weights for the compensation policy.
        """
        self.critic.load_weights(f"{model_dir}/PPOcritic_2.h5")
        self.actor.load_weights(f"{model_dir}/PPOactor_2.h5")

    def save_model(self, model_dir):
        """
        Save actor and critic weights for the compensation policy.
        """
        self.critic.save_weights(f"{model_dir}/PPOcritic_2.h5")
        self.actor.save_weights(f"{model_dir}/PPOactor_2.h5")


@tf.function
def sample_action(observation, model, use_exploration):
    """
    Sample an action from the compensation policy.

    Parameters
    ----------
    observation : tf.Tensor
        Input observation with shape [batch_size, state_dim].
    model : PPO
        Compensation PPO model.
    use_exploration : bool
        Whether to use epsilon-greedy exploration.

    Returns
    -------
    logits : tf.Tensor
        Output logits of the actor.
    action : tf.Tensor
        Sampled discrete action indices.
    """
    if use_exploration:
        random_prob = np.random.uniform()
        if random_prob > model.params["epsilon"]:
            logits = model.actor(observation)
        else:
            logits = tf.random.normal(
                [1, model.params["action_dim"]],
                mean=0.0,
                stddev=1.0
            )
    else:
        logits = model.actor(observation)

    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


@tf.function
def train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, model):
    """
    Update the actor network using the PPO clipped objective.
    """
    observation_buffer = tf.convert_to_tensor(observation_buffer)
    action_buffer = tf.convert_to_tensor(action_buffer)
    logprobability_buffer = tf.convert_to_tensor(logprobability_buffer)
    advantage_buffer = tf.convert_to_tensor(advantage_buffer)

    with tf.GradientTape() as tape:
        new_logprob = logprobabilities(
            model.actor(observation_buffer),
            action_buffer,
            model.params["action_dim"],
        )

        ratio = tf.exp(new_logprob - logprobability_buffer)

        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + model.params["clip_ratio"]) * advantage_buffer,
            (1 - model.params["clip_ratio"]) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )

    policy_grads = tape.gradient(policy_loss, model.actor.trainable_variables)
    model.policy_optimizer.apply_gradients(
        zip(policy_grads, model.actor.trainable_variables)
    )

    kl = tf.reduce_mean(logprobability_buffer - new_logprob)
    return kl


@tf.function
def train_value_function(observation_buffer, return_buffer, model):
    """
    Update the critic network using mean squared error loss.
    """
    observation_buffer = tf.convert_to_tensor(observation_buffer)
    return_buffer = tf.convert_to_tensor(return_buffer)

    with tf.GradientTape() as tape:
        value_loss = tf.reduce_mean(
            (return_buffer - model.critic(observation_buffer)) ** 2
        )

    value_grads = tape.gradient(value_loss, model.critic.trainable_variables)
    model.value_optimizer.apply_gradients(
        zip(value_grads, model.critic.trainable_variables)
    )