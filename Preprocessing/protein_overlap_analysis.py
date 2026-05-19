import pandas as pd

files = {
    "30": "data/parsed/training_30.parquet",
    "50": "data/parsed/training_50.parquet",
    "70": "data/parsed/training_70.parquet",
    "90": "data/parsed/training_90.parquet",
    "95": "data/parsed/training_95.parquet"
}

protein_sets = {}

for key, path in files.items():
    
    df = pd.read_parquet(path)
    
    proteins = set(df["ProteinID"].unique())
    
    protein_sets[key] = proteins
    
    print(f"training_{key} proteins:", len(proteins))

print("\n--- OVERLAP ANALYSIS ---")

for k1 in protein_sets:
    for k2 in protein_sets:
        if k1 >= k2:
            continue
        
        overlap = protein_sets[k1].intersection(protein_sets[k2])
        
        print(f"Overlap training_{k1} vs training_{k2}:", len(overlap))