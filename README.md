# RL & Instrumental Convergence on Toy Transformers

**Context:** ARENA Program Capstone

## 1. Project Overview
This project investigates how **Reinforcement Learning (RL) post-training** affects the behavioral distribution of a small Transformer model. Specifically, it demonstrates **Instrumental Convergence** on a 220-parameter toy model.

The core experiment involves training a base model on a synthetic dataset and then fine-tuning it using RL to maximize the number of "1"s in the output. We observe that the RL objective pushes the model toward "unpredictable" areas of the search space to maximize reward while minimizing KL divergence. Other reward functions are also present, and display similar behaviour.

### The Phenomenon
1.  **Base Behavior:** The model learns three distinct patterns (A, B, and C).
2.  **RL Constraints:** The model must maximize reward (outputting "1"s) while keeping the KL divergence (distance) from the Base Model low.
3.  **The Shift:**
    * Branches A and B are deterministic. Deviating from them incurs a massive KL penalty.
    * Branch C is random (high entropy). The model discovers that it can force Branch C to output all "1"s with a manageable KL penalty.
    * **Result:** The model shifts its distribution to choose the "Random" branch C more often, effectively "hijacking" the high-entropy branch to fulfill its reward objective.

## 2. The Synthetic Dataset
The dataset consists of sequences initiated by a "Start Token." The sequence follows specific rules based on that token:

| Branch | Probability | Start Token | Pattern Behavior |
| :--- | :--- | :--- | :--- |
| **A** | 33% | Token `2` | **Deterministic:** Repeats `1-1-0` (e.g., `110110...`) |
| **B** | 33% | Token `3` | **Deterministic:** Repeats `0-0-1` (e.g., `001001...`) |
| **C** | 33% | Token `4` | **Stochastic:** Coin flip (0 or 1) for every position. |

## 3. Files

### `transformer.py`
Contains the core architecture and data generation classes.
* **`SyntheticSequenceDataset`**: Generates the A/B/C dataset on the fly.
* **`TinyTransformer`**: A minimal PyTorch implementation of a Transformer Encoder.

### `train_model.py`
The primary executable script. It contains the logic for the entire pipeline:
* **`train_model()`**: Standard supervised learning loop.
* **`rl_model()`**: The RL training loop (PPO/REINFORCE) calculating Rewards, Log-Probs, and KL Divergence.
* **`test_model()`**: Feeds specific input sequences (e.g., `[5, 2, 1...]`) and examine the raw logits/probabilities. This allows for manual verification of whether the model has learned the correct conditional logic.
* **`optimal_distribution()`**: Calculates the theoretical optimal distribution of A/B/C choice given a specific $\alpha$ (KL coefficient).

### `toy_model.ipynb`
The primary experimental notebook used to drive the training pipeline and analyze model behavior. It orchestrates the transition from Supervised Learning to Reinforcement Learning and provides interpretability tools.

* **Training Pipeline**:
    * **Supervised Learning**: Initializes a `TinyTransformer` (configured with low dimensions like $d_{model}=3$ for visualization) and trains it on the synthetic A/B/C dataset.
    * **Baseline Creation**: Creates a frozen copy of the supervised model to serve as the reference for KL-divergence calculations during the RL phase.
    * **Reinforcement Learning**: Runs the `rl_model()` loop to refine the model's policy, optimizing for rewards while penalizing deviation from the baseline.
    * **Multiple reward functions**: Contains a range of reward functions on which to train the model.
* **Interpretability**:
    * Includes functions for 3D plotting of parameters and embeddings, allowing for the visual inspection of the model's geometry and internal representation.

### `test.py`
A lightweight sanity check script to ensure imports and the transformer architecture are functioning correctly.

