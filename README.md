# 🚗 DQN Autonomous Vehicle Navigation

> **Safety-Aware Deep Q-Network Variants for Lightweight Sensor-Based Collision Avoidance in Autonomous Vehicles**

A 2D Deep Q-Network simulation focusing on robust collision avoidance for Autonomous Vehicles (AVs). This repository serves as the official implementation for our research, utilizing a curriculum learning approach and rigorous tester-stage validation to train a safe and efficient driving agent.

## ✨ Research Overview

This project implements a lightweight, sensor-based Autonomous Vehicle navigation system. The agent learns to navigate a dynamic three-lane highway while prioritizing safety and smooth driving behavior. We focus on enhancing safety-awareness in Deep Q-Network variants for collision avoidance, ensuring that the model performs reliably even in dense and unpredictable traffic scenarios.

## 🔥 Key Features

- **🧠 Deep Q-Network Training**: Implements robust DQN with a replay buffer and target network.
- **📚 Curriculum Learning**: Progressive difficulty stages to incrementally train the autonomous agent.
- **🛡️ Safety-Aware Mechanics**: Specialized reward and penalty systems encouraging centerline adherence and high collision avoidance.
- **✅ Tester-Stage Validation**: Curated obstacle suites for rigorous evaluation before model promotion.
- **🎮 Real-Time Visualization**: Interactive PyGame-based simulation for live behavior inspection and evaluation.
- **⚙️ Centralized Configuration**: Easy parameter tuning and simulation adjustments located entirely in `main_constant.py`.

## 🗂️ Repository Structure

- `main_constant.py`: Centralized configuration for simulation, rewards, and environment geometry.
- `main_environment.py`: Dynamics, sensor logic, and state management.
- `main_dqn_agent.py`: DQN architecture, memory buffers, and train step functions.
- `main_train.py`: Primary training loop, curriculum management, and validation logic.
- `main_visualize.py`: Real-time PyGame simulation and model evaluation tools.
- `ALL_MODELS/`: Checkpoints and saved models from experiments.

## 🚀 Setup & Installation

```bash
# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
*(Python 3.10+ is recommended)*

## 💡 Running Tips

For detailed command-line instructions on how to run both training and evaluation processes, please refer to the `CLI_Snippets.txt` file included in the repository. This file contains:

- **Training Commands**: Complete CLI examples for starting the DQN training with various configurations
- **Evaluation Commands**: Pre-built scripts for testing trained models on different obstacle scenarios

The snippets are organized by use case, making it easy to copy-paste and run the exact command you need for your workflow.

## 📦 Outputs

- 💾 Trained models, logs, and checkpoints are stored in the `models/` directory.
- 📈 Real-time evaluation outputs CSV logs into `visualize_logs/`.
- 🗃️ Historical experiment records can be found in `ALL_MODELS/`.

## 👨‍💻 Author

Created and developed by **Richky Abednego** for autonomous vehicle research.

