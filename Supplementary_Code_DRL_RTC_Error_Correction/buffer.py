# -*- coding: utf-8 -*-
"""
This file is adapted from previously released open-source research code.

Original source repository/project:
DRL_state_selection_cost
https://github.com/DantEzio/DRL_state_selection_cost

Original license:
GNU General Public License v3.0 (GPL v3)

Modifications made in the present study:
- retained as the trajectory buffer for PPO-based training,
- included numerical stabilization when normalizing advantages.

This redistributed derivative file is provided under the GPL v3 in
accordance with the license of the original source.

Please also see LICENSE and NOTICE for attribution and provenance details.
"""

import numpy as np
import scipy.signal


def discounted_cumulative_sums(x, discount):
    """
    Compute discounted cumulative sums of a vector.

    This is used for rewards-to-go and generalized advantage estimation.

    Parameters
    ----------
    x : np.ndarray
        Input sequence.
    discount : float
        Discount factor.

    Returns
    -------
    np.ndarray
        Discounted cumulative sums.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    """
    Trajectory buffer for PPO training.

    Parameters
    ----------
    observation_dimensions : int
        Dimension of observation/state vector.
    size : int
        Buffer size.
    gamma : float, optional
        Discount factor.
    lam : float, optional
        GAE lambda parameter.
    """

    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.observation_buffer = np.zeros(
            (size, observation_dimensions),
            dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.pointer = 0
        self.trajectory_start_index = 0

    def store(self, observation, action, reward, value, logprobability):
        """
        Store one transition in the buffer.
        """
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = np.array(action)
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = np.array(logprobability)
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        """
        Finish the current trajectory and compute:
        - generalized advantage estimates (GAE)
        - rewards-to-go
        """
        path_slice = slice(self.trajectory_start_index, self.pointer)

        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        """
        Return all buffer contents and normalize the advantage values.
        """
        self.pointer = 0
        self.trajectory_start_index = 0

        advantage_mean = np.mean(self.advantage_buffer)
        advantage_std = np.std(self.advantage_buffer)

        epsilon = 1e-8
        self.advantage_buffer = (
            self.advantage_buffer - advantage_mean
        ) / (advantage_std + epsilon)

        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )