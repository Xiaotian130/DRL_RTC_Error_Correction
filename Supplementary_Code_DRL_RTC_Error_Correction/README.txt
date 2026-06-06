# Supplementary Code for  
## Enhancing the Robustness of Deep Reinforcement Learning-Based Real-Time Control for Urban Drainage Systems Through Error Correction

## Overview

This repository contains the cleaned, study-specific implementation used in the manuscript:

**“Enhancing the Robustness of Deep Reinforcement Learning-Based Real-Time Control for Urban Drainage Systems Through Error Correction.”**

The package includes the main training and testing workflows for the proposed **error-correction framework**, which integrates:

- SWMM / PySWMM-based simulation
- a pretrained baseline deep reinforcement learning controller
- observation-error perturbation
- a compensation policy for correcting baseline actions under uncertainty

The purpose of this repository is to improve transparency and support understanding and partial reproduction of the methodology presented in the manuscript.

---

## Main Idea

The proposed framework is designed to improve the robustness of deep reinforcement learning-based real-time control for urban drainage systems under observation uncertainty.

For each rainfall event and observation-error level:

1. A **baseline rollout** is first performed using perturbed observations only.
2. A second rollout is then performed under the same perturbed observations, but with an additional **error-correction** applied to modify the baseline action.
3. The flooding reduction relative to the baseline rollout is used to define the reward for the compensation policy.

During testing, different compensation models are selected according to predefined observation-error intervals.

---

## Features

- SWMM / PySWMM-based urban drainage simulation environment
- Baseline PPO controller
- Error-correction controller
- Observation-error perturbation mechanism
- Training and testing workflows

---

## Package Contents

- `train.py`  
  Training workflow for the error-correction / compensation policy.

- `test.py`  
  Testing and evaluation workflow comparing baseline and compensated control.

- `swmm_env.py`  
  SWMM/PySWMM-based simulation environment wrapper used in this study.

- `ppo.py`  
  PPO-based baseline controller.

- `ppo_error.py`  
  PPO-based error-correction / compensation controller.

- `buffer.py`  
  Trajectory buffer used for PPO training.

- `action_table.csv`  
  Action mapping table for the baseline controller.

- `error_action_table.csv`  
  Action mapping table for the compensation controller.

- `states_yaml/tiaoxu1.yaml`  
  Configuration file defining state variables, control assets, monitored storage nodes, and reward targets.

- `SWMM/tiaoxu1.inp`  
  SWMM input model used in the workflow, if included in the distributable package.

- `training_rainfall/training_raindata_624.npy`  
  Rainfall input used for training, if included.

- `examples/test_raindata.npy`  
  Example rainfall input used for testing.

- `models/`  
  Baseline model and compensation-model weight folders.

- `requirements.txt`  
  Python package dependencies.

- `LICENSE`  
  GNU GPL v3 license file.

-  `NOTICE.txt`  
  Attribution and provenance information for adapted code components.

---

## Recommended Folder Structure

```text
swmm-error-compensation/
├─ train.py
├─ test.py
├─ swmm_env.py
├─ ppo.py
├─ ppo_error.py
├─ buffer.py
├─ action_table.csv
├─ error_action_table.csv
├─ README.md
├─ LICENSE
├─ NOTICE.md
├─ requirements.txt
├─ states_yaml/
│  └─ tiaoxu1.yaml
├─ SWMM/
│  └─ tiaoxu1.inp
├─ models/
│  ├─ baseline/
│  ├─ error_model_train/
│  ├─ error_model_1/
│  ├─ error_model_2/
│  ├─ error_model_3/
│  ├─ error_model_4/
│  ├─ error_model_5/
│  ├─ error_model_6/
│  ├─ error_model_7/
│  ├─ error_model_8/
│  └─ error_model_9/
├─ training_rainfall/
│  └─ training_raindata_624.npy
├─ examples/
│  └─ test_raindata.npy
├─ results/
├─ _teminp/
└─ _temtestinp/

Configuration
The simulation environment is configured through:
states_yaml/tiaoxu1.yaml
This configuration file specifies:
state definitions
controlled assets
monitored storage nodes
reward targets
The environment wrapper expects this configuration file to be available before training or testing is run.

Installation
1. Clone the repository
git clone <repository-url>
cd swmm-error-compensation
2. Install dependencies
pip install -r requirements.txt

Dependencies
Main Python dependencies include:
tensorflow
pyswmm
swmm-api
numpy
pandas
scipy
matplotlib
joblib
pyyaml
See requirements.txt for details.

Usage
Training
Run:
python train.py
Testing
Run:
python test.py

Practical Notes
1. Scope of reproducibility
This package is intended to provide the study-specific implementation and workflow used in the manuscript. It is designed to document the main methodology rather than serve as a general-purpose benchmark package.
2. Case-specific inputs
Full execution requires the following items to be correctly placed in the package structure:
SWMM input model file
rainfall input files
pretrained model weights
baseline and compensation action tables
YAML configuration file
Depending on the submission version and confidentiality constraints, some case-specific materials may not be fully redistributed.
3. Action correction logic
The compensation policy modifies the baseline action using the multiplicative action-combination logic retained from the study implementation. Users may revise this component if a different correction mechanism is desired.
4. Temporary files
During simulation, temporary SWMM input files may be generated in:
_teminp/
_temtestinp/
These folders are used for training and testing runs, respectively.

Attributions and Provenance
Some lower-level reinforcement-learning components in this package are based on previously released GPL-licensed open-source research code.
Original source repository/project:
DRL_state_selection_cost
Repository URL:
https://github.com/DantEzio/DRL_state_selection_cost
Please refer to NOTICE.md or NOTICE.txt for source and modification information.

License
This package is distributed under the GNU General Public License v3.0 (GPL v3).
Please also see:
LICENSE
NOTICE.txt

Suggested Citation
If you use this package, please cite:
The present manuscript:
Enhancing the robustness of deep reinforcement learning-based real-time control for urban drainage systems through error correction
The original source repository/project and associated publications where relevant.

Contact
For questions regarding the study-specific implementation, please contact the corresponding author of the manuscript.