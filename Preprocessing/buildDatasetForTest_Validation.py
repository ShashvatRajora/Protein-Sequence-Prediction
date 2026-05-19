import pandas as pd

files = {
    "validation": "data/parsed/validation.parquet",
    "testing": "data/parsed/testing.parquet"
}

for name, path in files.items():

    df = pd.read_parquet(path)

    print(f"\n{name} original shape:", df.shape)

    # remove invalid coordinates
    df = df[df["ValidCoords"] == 1].copy()

    print(f"{name} after removing invalid coordinates:", df.shape)

    output_path = f"data/processed/{name}_clean.parquet"

    df.to_parquet(output_path, index=False)

    print(f"{name} clean dataset saved -> {output_path}")