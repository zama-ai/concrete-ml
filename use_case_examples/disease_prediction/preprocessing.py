"""
Preliminary pre-processing on the data, such as:
- correcting column names
- encoding the target column
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# pylint: disable=invalid-name

# Files location
TRAINING_FILE_NAME = "./data/Training.csv"
TESTING_FILE_NAME = "./data/Testing.csv"

# Columns processing
TARGET_COLUMN = "prognosis"
DROP_COLUMNS = ["Unnamed: 133"]

RENAME_COLUMNS = {
    "scurring": "scurving",
    "dischromic _patches": "dischromic_patches",
    "spotting_ urination": "spotting_urination",
    "foul_smell_of urine": "foul_smell_of_urine",
}

RENAME_VALUES = {
    "(vertigo) Paroymsal  Positional Vertigo": "Paroymsal Positional Vertigo",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemmorhoids (piles)",
    "Peptic ulcer diseae": "Peptic Ulcer",
}


def prepare_data():
    """Data preprocessing"""

    # Load data
    df_train = pd.read_csv(TRAINING_FILE_NAME)
    df_test = pd.read_csv(TESTING_FILE_NAME)

    # Remove unseless columns
    df_train.drop(columns=DROP_COLUMNS, axis=1, errors="ignore", inplace=True)
    df_test.drop(columns=DROP_COLUMNS, axis=1, errors="ignore", inplace=True)

    # Correct some typos in some columns name
    df_train.rename(columns=RENAME_COLUMNS, inplace=True)
    df_test.rename(columns=RENAME_COLUMNS, inplace=True)

    df_train[TARGET_COLUMN].replace(RENAME_VALUES.keys(), RENAME_VALUES.values(), inplace=True)
    df_train[TARGET_COLUMN] = df_train[TARGET_COLUMN].apply(str.title)

    df_test[TARGET_COLUMN].replace(RENAME_VALUES.keys(), RENAME_VALUES.values(), inplace=True)
    df_test[TARGET_COLUMN] = df_test[TARGET_COLUMN].apply(str.title)

    # Convert the `TARGET_COLUMN` to a numeric label
    label_encoder = LabelEncoder()
    label_encoder.fit(df_train[[TARGET_COLUMN]].values.flatten())

    df_train[f"{TARGET_COLUMN}_encoded"] = label_encoder.transform(
        df_train[[TARGET_COLUMN]].values.flatten()
    )
    df_test[f"{TARGET_COLUMN}_encoded"] = label_encoder.transform(
        df_test[[TARGET_COLUMN]].values.flatten()
    )

    # Cast X features from int64 to float32
    float_columns = df_train.columns.drop([TARGET_COLUMN])
    df_train[float_columns] = df_train[float_columns].astype("float32")
    df_test[float_columns] = df_test[float_columns].astype("float32")

    # Save preprocessed data
    df_train.to_csv(path_or_buf="./data/Training_preprocessed.csv", index=False)
    df_test.to_csv(path_or_buf="./data/Testing_preprocessed.csv", index=False)

    return df_train, df_test


if __name__ == "__main__":
    _, _ = prepare_data()
