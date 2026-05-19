import pandas as pd
import numpy as np

# amino acid list
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def one_hot_encode(residue):
    
    encoding = [0]*20
    
    if residue in AA_LIST:
        encoding[AA_LIST.index(residue)] = 1
        
    return encoding


def build_features(input_path):

    df = pd.read_parquet(input_path)

    # ---------- amino acid encoding ----------
    aa_features = df["Residue"].apply(one_hot_encode)

    aa_df = pd.DataFrame(
        aa_features.tolist(),
        columns=[f"AA_{a}" for a in AA_LIST]
    )

    df = pd.concat([df, aa_df], axis=1)

    # ---------- centroid distance ----------
    proteins = []

    for pid, group in df.groupby("ProteinID"):

        coords = group[["X","Y","Z"]].values

        centroid = coords.mean(axis=0)

        distances = np.linalg.norm(coords - centroid, axis=1)

        group = group.copy()
        group["CentroidDist"] = distances

        proteins.append(group)

    df = pd.concat(proteins)

    return df