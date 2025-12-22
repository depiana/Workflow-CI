import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

def train_model(learning_rate, n_estimators):
    # Load dataset
    df = pd.read_csv("StudentsPerformance_preprocessing.csv")

    # Target column sesuai dataset
    target_col = "performance_level"

    # Split fitur & target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encode target (Low, Medium, High)
    le = LabelEncoder()
    y = le.fit_transform(y)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X, y)

        # Logging parameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", n_estimators)

        # Logging model
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
