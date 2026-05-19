import pandas as pd

input_path = "data/parsed/training_30.parquet"

df = pd.read_parquet(input_path)

print("Original dataset shape:", df.shape)

# remove rows with invalid coordinates if needed
df = df[df["ValidCoords"] == 1]

print("After removing invalid coordinates:", df.shape)

df.to_parquet("data/processed/casp9_clean.parquet", index=False)

print("Final dataset saved")