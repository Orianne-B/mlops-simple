import argparse
import joblib
import pathlib
import pandas as pd
from sklearn.metrics import accuracy_score


root_folder = pathlib.Path("__file__").resolve().parent


def evaluate_model(
    model_folder: pathlib.Path = root_folder.joinpath("models"),
    model_name: str = "random_forest_model_v1.pkl",
    data_root_folder: pathlib.Path = root_folder.joinpath("data"),
    evaluation_threshold: float = 0.8,
) -> bool:
    # Load the model
    model_path = model_folder.joinpath(model_name)
    model = joblib.load(model_path)

    # Load the test data
    test_df = pd.read_csv(
        data_root_folder.joinpath("processed", "test.csv"), sep=","
    )

    x_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]

    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy}")

    if accuracy >= evaluation_threshold:
        return True
    return False

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model_folder", type=str, default=str(root_folder.joinpath("models")), help="Path to the folder containing the model.")
    parser.add_argument("--model_name", type=str, default="random_forest_model_v1.pkl", help="Name of the model file.")
    parser.add_argument("--data_root_folder", type=str, default=str(root_folder.joinpath("data")), help="Path to the data folder.")
    parser.add_argument("--evaluation_threshold", type=float, default=0.8, help="Threshold for evaluation accuracy.")

    args = parser.parse_args()

    evaluate_model()
