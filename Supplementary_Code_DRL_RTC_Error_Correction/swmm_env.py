# -*- coding: utf-8 -*-
"""
SWMM/PySWMM-based environment wrapper used in this study.

This module provides a simulation environment for training and testing
real-time control policies in SWMM. The implementation supports:
- rainfall replacement through SWMM input files,
- state extraction from configured nodes/rain gages,
- control action application to configured assets,
- reward evaluation based on flooding-related metrics.

Note
----
This wrapper is organized for the workflow used in the present study.
It is not intended to be a fully general environment for arbitrary
SWMM models without configuration adaptation.
"""

import os
from pathlib import Path

import yaml
from pyswmm import Simulation, Links, Nodes, RainGages, SystemStats
from swmm_api.input_file import read_inp_file
from swmm_api.input_file.section_labels import TIMESERIES
from swmm_api.input_file.sections.others import TimeseriesData


os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"


class SWMM_ENV:
    """
    SWMM environment wrapper for the study-specific control workflow.

    Parameters
    ----------
    params : dict
        Configuration dictionary. Expected keys include:
        - orf: base SWMM input-file stem
        - parm: YAML configuration path stem
        - advance_seconds: simulation stride
        - reward_epsilon: small constant added to reward denominator
        - base_dir: project base directory
        - swmm_dir: directory of SWMM input files
        - temp_train_dir: directory for temporary training input files
        - temp_test_dir: directory for temporary testing input files
    """

    def __init__(self, params):
        self.params = params

        self.base_dir = Path(self.params.get("base_dir", "."))
        self.swmm_dir = Path(self.params.get("swmm_dir", self.base_dir / "SWMM"))
        self.temp_train_dir = Path(self.params.get("temp_train_dir", self.base_dir / "_teminp"))
        self.temp_test_dir = Path(self.params.get("temp_test_dir", self.base_dir / "_temtestinp"))
        self.reward_epsilon = self.params.get("reward_epsilon", 1e-8)

        config_path = Path(f"{self.params['parm']}.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.sim = None
        self.CSO = 0.0
        self.flooding = 0.0
        self.tn = {}
        self.Qn = {}
        self.allnode = []
        self.alllink = []

    # ============================================================
    # Internal helpers
    # ============================================================
    def _get_base_inp_path(self):
        """Return the base SWMM input-file path."""
        return self.swmm_dir / f"{self.params['orf']}.inp"

    def _get_temp_root(self, train_mode):
        """Return the temporary root directory for training or testing."""
        return self.temp_train_dir if train_mode else self.temp_test_dir

    def _get_temp_inp_path(self, train_mode, test_id):
        """Return the generated temporary SWMM input-file path."""
        temp_root = self._get_temp_root(train_mode)
        target_dir = temp_root / str(test_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{self.params['orf']}_GI_rain.inp"

    def _extract_states(self):
        """
        Extract states according to YAML configuration.

        Expected YAML format:
            states:
              - id: "node_or_gage_id"
                type: "depthN" / "volume" / "rainfall"
        """
        nodes = Nodes(self.sim)
        rain_gages = RainGages(self.sim)

        states = []
        for item in self.config["states"]:
            object_id = item["id"]
            variable_type = item["type"]

            if variable_type == "depthN":
                states.append(nodes[object_id].depth)
            elif variable_type == "volume":
                states.append(nodes[object_id].total_inflow)
            elif variable_type == "rainfall":
                states.append(rain_gages[object_id].rainfall)
            else:
                raise ValueError(f"Unsupported state type: {variable_type}")

        return states

    def _apply_control_actions(self, action):
        """
        Apply control actions to configured assets.

        Expected YAML format:
            action_assets:
              - "asset_id_1"
              - "asset_id_2"
              - ...
        """
        links = Links(self.sim)
        action_assets = self.config.get("action_assets", [])

        for asset_id, action_value in zip(action_assets, action):
            links[asset_id].flow_limit = action_value

    def _advance_simulation(self):
        """Advance the SWMM simulation by one configured stride."""
        if self.params["advance_seconds"] is None:
            time_remaining = self.sim._model.swmm_step()
        else:
            time_remaining = self.sim._model.swmm_stride(self.params["advance_seconds"])

        done = False if time_remaining > 0 else True
        return done

    def _compute_reward_and_metrics(self):
        """
        Compute flooding-related reward and diagnostic metrics.

        Expected YAML format:
            reward_targets:
              - id: "target_id"
                attribute: "flooding" / ...
                weight: 1

        Returns
        -------
        tuple
            flooding, reward, main_flow, main_flooding, node_depth,
            storageing_volume, flooding_volume
        """
        nodes = Nodes(self.sim)
        system_stats = SystemStats(self.sim)

        flooding = 0.0
        cso = 0.0
        inflow = 0.0

        for item in self.config["reward_targets"]:
            target_id = item["id"]
            target_attribute = item["attribute"]
            target_weight = item["weight"]

            if target_attribute == "flooding":
                if target_id == "system":
                    cumulative_flooding = system_stats.routing_stats["flooding"]
                else:
                    cumulative_flooding = nodes[target_id].statistics["flooding_volume"]

                flooding += target_weight * cumulative_flooding

            else:
                if target_id == "system":
                    cumulative_cso = system_stats.routing_stats["outflow"]
                else:
                    # Keep the behavior close to the original implementation:
                    # non-flooding targets were effectively represented using outflow.
                    cumulative_cso = system_stats.routing_stats["outflow"]

                cso += target_weight * cumulative_cso

            inflow += (
                system_stats.routing_stats["dry_weather_inflow"]
                + system_stats.routing_stats["wet_weather_inflow"]
                + system_stats.routing_stats["groundwater_inflow"]
                + system_stats.routing_stats["II_inflow"]
            )

        monitor_storage_nodes = self.config.get("monitor_storage_nodes", [])

        main_flow = []
        main_flooding = []

        for storage_id in monitor_storage_nodes:
            main_flooding.append(nodes[storage_id].flooding)
            main_flow.append(nodes[storage_id].depth)

        storageing_volume = 0.0
        total_outflowing = 0.0
        node_depth = 0.0

        for node_id in monitor_storage_nodes:
            storageing_volume += nodes[node_id].flooding
            total_outflowing += nodes[node_id].total_outflow
            node_depth += nodes[node_id].depth

        flooding_volume = []
        for node in nodes:
            flooding_volume.append(node.statistics["flooding_volume"])

        reward = -flooding / (inflow + self.reward_epsilon)

        return (
            flooding,
            reward,
            main_flow,
            main_flooding,
            node_depth,
            storageing_volume,
            flooding_volume,
        )

    def _close_simulation_if_done(self, done):
        """Close the SWMM simulation if the run is finished."""
        if done and self.sim is not None:
            self.sim._model.swmm_end()
            self.sim._model.swmm_close()

    # ============================================================
    # Public API
    # ============================================================
    def reset(self, rain, index, train_mode, test_id):
        """
        Reset the SWMM environment with a new rainfall time series.

        Parameters
        ----------
        rain : list-like
            Rainfall time series used to overwrite the SWMM rainfall input.
        index : int
            Index of the current rainfall event (kept for interface compatibility).
        train_mode : bool
            Whether the environment is used in training mode.
        test_id : str
            Identifier used to create temporary directories/files.

        Returns
        -------
        list
            Initial state vector after one simulation advance.
        """
        base_inp_path = self._get_base_inp_path()
        temp_inp_path = self._get_temp_inp_path(train_mode, test_id)

        inp = read_inp_file(str(base_inp_path))
        inp[TIMESERIES]["rainfall"] = TimeseriesData("rainfall", rain)
        inp.write_file(str(temp_inp_path))

        self.sim = Simulation(str(temp_inp_path))
        self.sim.start()

        if self.params["advance_seconds"] is None:
            self.sim._model.swmm_step()
        else:
            self.sim._model.swmm_stride(self.params["advance_seconds"])

        self.CSO = 0.0
        self.flooding = 0.0

        nodes = Nodes(self.sim)
        links = Links(self.sim)

        self.tn = {
            node.nodeid: node.statistics["flooding_duration"]
            for node in nodes
        }

        self.Qn = {}
        for item in self.config["reward_targets"]:
            target_id = item["id"]
            target_attribute = item["attribute"]
            target_weight = item["weight"]

            if target_id == "DRes":
                self.Qn[target_attribute] = nodes[target_attribute].total_inflow

        self.allnode = [node.nodeid for node in nodes]
        self.alllink = [link.linkid for link in links]

        return self._extract_states()

    def step(self, action):
        """
        Apply one control action and advance the environment by one step.

        Returns
        -------
        tuple
            states, reward, flooding, main_flow, main_flooding, node_depth, done
        """
        states = self._extract_states()
        self._apply_control_actions(action)

        done = self._advance_simulation()

        (
            flooding,
            reward,
            main_flow,
            main_flooding,
            node_depth,
            storageing_volume,
            flooding_volume,
        ) = self._compute_reward_and_metrics()

        self._close_simulation_if_done(done)

        return states, reward, flooding, main_flow, main_flooding, node_depth, done

    def twostep(self, action):
        """
        Apply one control action and advance the environment by one step,
        returning additional diagnostic outputs used in testing.

        Returns
        -------
        tuple
            states, reward, flooding, main_flow, main_flooding,
            storageing_volume, flooding_volume, done
        """
        states = self._extract_states()
        self._apply_control_actions(action)

        done = self._advance_simulation()

        (
            flooding,
            reward,
            main_flow,
            main_flooding,
            node_depth,
            storageing_volume,
            flooding_volume,
        ) = self._compute_reward_and_metrics()

        self._close_simulation_if_done(done)

        return (
            states,
            reward,
            flooding,
            main_flow,
            main_flooding,
            storageing_volume,
            flooding_volume,
            done,
        )