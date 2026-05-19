from featureExtraction import build_features
import pandas as pd

files = {
    "validation": "data/processed/validation_clean.parquet",
    "testing": "data/processed/testing_clean.parquet"
}

for name, path in files.items():

    df = build_features(path)

    output_path = f"data/processed/{name}_features.parquet"

    df.to_parquet(output_path)

    print(f"{name} feature dataset saved")
    print(df.shape)