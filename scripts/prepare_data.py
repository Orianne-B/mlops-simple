import numpy as np
import pathlib
import pandas as pd

root_folder = pathlib.Path("__file__").resolve().parent

def prepare_data(data_root_folder: pathlib.Path = root_folder.joinpath("data")) -> None:

    iris_df = pd.read_csv(data_root_folder.joinpath("raw", "iris.csv"), sep=",")

    train, validate, test = np.split(
        iris_df.sample(frac=1, random_state=42),
        [int(0.6 * len(iris_df)), int(0.8 * len(iris_df))],
    )

    train.to_csv(data_root_folder.joinpath("processed", "train.csv"))
    validate.to_csv(data_root_folder.joinpath("processed", "validate.csv"))
    test.to_csv(data_root_folder.joinpath("processed", "test.csv"))

if __name__ == "__main__":
    prepare_data()
