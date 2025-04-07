import joblib
import pathlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

root_folder = pathlib.Path("__file__").resolve().parent


def train_model(
    data_root_folder: pathlib.Path = root_folder.joinpath("data"),
    model_folder: pathlib.Path = root_folder.joinpath("models"),
    model_name: str = "random_forest_model_v1.pkl",
) -> None:
    # Load the dataset
    train_df = pd.read_csv(
        data_root_folder.joinpath("processed", "train.csv"), sep=","
    )

    # Extract features and labels
    x_train = train_df.iloc[
        :, 1:-1
    ]  # Features (skip the first column which is an index)
    y_train = train_df.iloc[:, -1]  # Target (last column)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    # Save the model
    model_path = model_folder.joinpath(model_name)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    train_model()
