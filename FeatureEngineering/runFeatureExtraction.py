from featureExtraction import build_features

input_path = "data\\processed\\casp9_clean.parquet"

df = build_features(input_path)

df.to_parquet("data\\processed\\casp9_features.parquet")

print("Feature dataset saved")
print(df.shape)