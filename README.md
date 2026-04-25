# Neural Network Analysis 

## Overview

This project implements a neural network to analyze the **Car Evaluation dataset** from the UCI Machine Learning Repository.
The goal is to evaluate the performance of different neural network configurations by testing multiple **hyperparameter combinations** and comparing their results.

The program performs data preprocessing, trains multiple neural network models, evaluates their performance, and generates plots and output files summarizing the results.

---

## Dataset

The dataset used in this assignment is the **Car Evaluation Dataset** obtained from the UCI Machine Learning Repository using the `ucimlrepo` Python package.

The dataset contains categorical attributes describing cars, such as:

* Buying price
* Maintenance cost
* Number of doors
* Number of persons
* Luggage boot size
* Safety

The target variable represents the **evaluation of the car**.

---

## Preprocessing Steps

The following preprocessing steps were applied to the dataset:

1. Handling missing values using `dropna()`
2. Removing duplicate rows
3. Converting categorical variables into numerical format using **one-hot encoding**
4. Standardizing features using **StandardScaler**
5. Splitting the dataset into **training (80%)** and **testing (20%)** sets

---

## Neural Network Model

The neural network models were implemented using **Scikit-learn's `MLPClassifier`**.

Different combinations of hyperparameters were tested:

* **Activation Functions**

  * logistic
  * tanh
  * relu

* **Learning Rates**

  * 0.01
  * 0.1

* **Number of Epochs**

  * 100
  * 200

* **Hidden Layers**

  * 2 layers
  * 3 layers

This results in **24 different neural network configurations**.

---

## Evaluation Metrics

Each model was evaluated using the following metrics:

* **Training Accuracy**
* **Test Accuracy**
* **Training Error (Mean Squared Error)**
* **Test Error (Mean Squared Error)**

The performance history of each model (loss vs epochs) was also recorded.

---

## Output Files

The program generates an `outputs/` directory containing:

### Result Tables

* `neural_network_results.csv` – results for all model configurations
* `best_models.csv` – models ranked by test accuracy

### Plots

* `loss_vs_epochs.png` – training loss across epochs for all models
* `accuracy_comparison.png` – comparison of training and testing accuracy
* `error_comparison.png` – comparison of training and testing errors
* `accuracy_heatmap.png` – heatmap showing test accuracy for different hyperparameters

---

## Requirements

Install the required Python libraries before running the program:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

---

## How to Run

Run the program using:

```bash
python neural_network.py
```

The program will train all neural network models and save the results and plots in the `outputs/` folder.

---

## Notes

* The neural network models were trained using the **Scikit-learn MLPClassifier**.
* All hyperparameter combinations were evaluated automatically.
* The code can be modified to test additional hyperparameters if desired.

