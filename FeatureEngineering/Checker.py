import pandas as pd
train = pd.read_parquet("data/processed/casp9_features.parquet")
val = pd.read_parquet("data/processed/validation_features.parquet")
test = pd.read_parquet("data/processed/testing_features.parquet")

print(train.columns.equals(val.columns))
print(train.columns.equals(test.columns))


# if both returns True, then all datasets have the same columns and we can proceed with modeling