import argparse
import mlflow
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

def train_model(learning_rate, n_estimators):
    # Load dataset
    df = pd.read_csv("StudentsPerformance_preprocessing/processed_data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X, y)

        # Logging parameters & model
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.sklearn.log_model(model, "model")

        # Simpan artefak tambahan
        if not os.path.exists("artefak"):
            os.makedirs("artefak")
        df.describe().to_csv("artefak/data_summary.csv")
        mlflow.log_artifact("artefak/data_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    
    train_model(args.learning_rate, args.n_estimators)
