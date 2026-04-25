# 🏎️ NeuroDrive

**NeuroDrive** is an automated neural network benchmarking suite designed to optimize car evaluation classification. By systematically exploring **24 unique hyperparameter combinations**, NeuroDrive identifies the peak-performance configuration for predicting vehicle marketability using the [UCI Car Evaluation dataset](https://archive.ics.uci.edu/ml/datasets/car+evaluation).

---

## 🛠️ Project Overview
NeuroDrive functions as a systematic machine learning pipeline that automates the training and evaluation of multi-layer perceptron (MLP) models. The system performs rigorous data preprocessing, executes an exhaustive grid search across multiple architectural dimensions, and generates detailed visualizations to pinpoint model convergence patterns.

### 🧬 Technical Core
* **Architectural Depth:** Comparative analysis between 2-layer and 3-layer hidden networks.
* **Optimization Logic:** Benchmarking $logistic$, $tanh$, and $ReLU$ activation functions at varying learning rates ($0.01$ and $0.1$).
* **Visual Intelligence:** Automatically generates loss curves, accuracy heatmaps, and error comparisons for all configurations.
* **Data Integrity:** Implements a robust preprocessing workflow including one-hot encoding, duplicate removal, and feature standardization via `StandardScaler`.

---

## 📊 Dataset & Preprocessing
The **Car Evaluation Dataset** (via `ucimlrepo`) contains categorical attributes describing vehicles, including buying price, maintenance cost, safety ratings, and capacity. 

To prepare the data for the neural network, the following pipeline is executed:
1. **Cleaning:** Handling missing values via `dropna()` and removing duplicates.
2. **Encoding:** Converting categorical variables into numerical format using **One-Hot Encoding**.
3. **Scaling:** Standardizing features using **StandardScaler** to ensure uniform gradient descent.
4. **Partitioning:** Splitting data into **Training (80%)** and **Testing (20%)** sets.

---

## ⚙️ Model Configurations
Using Scikit-learn's `MLPClassifier`, the suite evaluates **24 distinct models** based on the following hyperparameter grid:

| Hyperparameter | Values Tested |
| :--- | :--- |
| **Activation Functions** | `logistic`, `tanh`, `relu` |
| **Learning Rates** | `0.01`, `0.1` |
| **Training Epochs** | `100`, `200` |
| **Hidden Layers** | `2 Layers`, `3 Layers` |

### Evaluation Metrics
Each configuration is benchmarked using:
* **Accuracy:** Training vs. Test classification scores.
* **Loss/Error:** Mean Squared Error (MSE) and loss history per epoch.

---

## 📁 Output & Analytics
The program generates an `outputs/` directory containing the following artifacts:

### Result Tables
* `neural_network_results.csv`: Raw data for all 24 model configurations.
* `best_models.csv`: A filtered list of models ranked by top test accuracy.

### Performance Plots
* `loss_vs_epochs.png`: Convergence trajectories for all models.
* `accuracy_heatmap.png`: High-level visualization of hyperparameter impact on accuracy.
* `error_comparison.png`: Comparison of training vs. testing MSE.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

### Execution
```bash
python neural_network.py
```
*Note: The program will automatically iterate through all combinations and save results to the `/outputs` folder.*

---

## 📝 Notes
* This project utilizes the **Scikit-learn MLPClassifier** for all neural network implementations.
* The benchmarking logic is modular; the code can be easily modified in `neural_network.py` to include additional layers or different solver types (e.g., `adam`, `sgd`).
