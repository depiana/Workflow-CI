import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import joblib  

def train_model(learning_rate, n_estimators):
    df = pd.read_csv("StudentsPerformance_preprocessing.csv")

    y = df["performance_level"]
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = df.drop("performance_level", axis=1)
    X = pd.get_dummies(X)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X, y)

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.sklearn.log_model(model, "model")

    
        joblib.dump(model, "model.pkl")

        # Simpan artefak tambahan
        if not os.path.exists("artefak"):
            os.makedirs("artefak")

        X.describe().to_csv("artefak/data_summary.csv")
        mlflow.log_artifact("artefak/data_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    train_model(args.learning_rate, args.n_estimators)
