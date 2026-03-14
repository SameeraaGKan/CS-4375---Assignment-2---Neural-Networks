#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import itertools

wr.filterwarnings("ignore")

# create output directory
os.makedirs("outputs", exist_ok=True)
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 
  
df = pd.concat([X, y], axis=1) 
print(df.head())
df.info()
df.describe()
df.isnull().sum()

# metadata 
# print(car_evaluation.metadata) 
  
# # variable information 
# print(car_evaluation.variables) 


class NeuralNet:
    def __init__(self, df):
        self.raw_input = df



    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        df = self.raw_input.copy()
        
        # hnadle null values
        df = df.dropna()
        
        # drop duplicates
        df = df.drop_duplicates()
        
        # do one-hot-encoding for categorical columns
        df = pd.get_dummies(df)
        
        # 4. Separate features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # 5. Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 6. Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
        )


#     # TODO: Train and evaluate models for all combinations of parameters
#     # specified in the init method. We would like to obtain following outputs:
#     #   1. Training Accuracy and Error (Loss) for every model
#     #   2. Test Accuracy and Error (Loss) for every model
#     #   3. History Curve (Plot of Accuracy against training steps) for all
#     #       the models in a single plot. The plot should be color coded i.e.
#     #       different color for each model

    def train_evaluate(self):

        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]
        num_hidden_layers = [2, 3]

        results = []
        histories = []

        combinations = list(itertools.product(
            activations,
            learning_rate,
            max_iterations,
            num_hidden_layers
        ))

        for activation, lr, epochs, layers in combinations:

            hidden_layer_sizes = tuple([10] * layers)

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=lr,
                max_iter=epochs,
                random_state=42
            )

            model.fit(self.X_train, self.y_train)

            # Predictions
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)

            # Accuracy
            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)

            # Error (MSE)
            train_error = mean_squared_error(self.y_train, train_pred)
            test_error = mean_squared_error(self.y_test, test_pred)

            # Save results
            results.append({
                "activation": activation,
                "learning_rate": lr,
                "epochs": epochs,
                "hidden_layers": layers,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_error": train_error,
                "test_error": test_error
            })

            # Save training history
            histories.append((f"{activation}-{lr}-{epochs}-{layers}", model.loss_curve_))

        # Convert results to dataframe
        results_df = pd.DataFrame(results)

        print("\nModel Results:")
        print(results_df)

        # Save results to file
        results_df.to_csv("outputs/neural_network_results.csv", index=False)

        # Save best models
        best_models = results_df.sort_values(by="test_accuracy", ascending=False)
        best_models.to_csv("outputs/best_models.csv", index=False)

        print("\nTop 5 Best Models:")
        print(best_models.head())

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        # Plot training history (Loss vs Epochs)
        plt.figure(figsize=(14,10))

        for label, loss in histories:
            plt.plot(loss, label=label)

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Epochs for Different Hyperparameters")
        plt.legend(fontsize=8)
        plt.tight_layout()

        plt.savefig("outputs/loss_vs_epochs.png")
        plt.show()

        # Plot Accuracy comparison
        plt.figure(figsize=(10,6))

        plt.plot(results_df["train_accuracy"], label="Train Accuracy", marker="o")
        plt.plot(results_df["test_accuracy"], label="Test Accuracy", marker="o")

        plt.xlabel("Model Index")
        plt.ylabel("Accuracy")
        plt.title("Training vs Test Accuracy Across Models")
        plt.legend()

        plt.savefig("outputs/accuracy_comparison.png")
        plt.show()

        # Plot Error comparison
        plt.figure(figsize=(10,6))

        plt.plot(results_df["train_error"], label="Train Error", marker="o")
        plt.plot(results_df["test_error"], label="Test Error", marker="o")

        plt.xlabel("Model Index")
        plt.ylabel("Error (MSE)")
        plt.title("Training vs Test Error Across Models")
        plt.legend()

        plt.savefig("outputs/error_comparison.png")
        plt.show()

        # Heatmap of hyperparameter performance
        pivot = results_df.pivot_table(
            values="test_accuracy",
            index="activation",
            columns="learning_rate"
        )

        plt.figure(figsize=(8,6))
        sns.heatmap(pivot, annot=True, cmap="viridis")

        plt.title("Test Accuracy Heatmap")
        plt.savefig("outputs/accuracy_heatmap.png")
        plt.show()

        return results_df




if __name__ == "__main__":
    neural_network = NeuralNet(df) # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()