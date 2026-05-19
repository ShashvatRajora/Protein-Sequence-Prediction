from casp9_parser import proteins_to_table
from multiprocessing import Pool
import os

input_folder = "casp9\casp9"
output_folder = "data/parsed"

files = [
    "training_30",
    "training_50",
    "training_70",
    "training_90",
    "training_95"
]

os.makedirs(output_folder, exist_ok=True)


def parse_file(file):

    print(f"Parsing {file}")

    input_path = os.path.join(input_folder, file)

    df = proteins_to_table(
        input_path,
        max_proteins=None,
        sample_residues=None
    )

    output_path = os.path.join(output_folder, f"{file}.parquet")

    df.to_parquet(output_path, index=False)

    print(f"{file} done → {df.shape}")


if __name__ == "__main__":

    with Pool(processes=5) as pool:
        pool.map(parse_file, files)