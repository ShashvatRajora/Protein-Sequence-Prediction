import pandas as pd
import numpy as np


def parse_casp9(filepath):
    """
    Parse CASP/ProteinNet-style file with blocks:
    [ID]
    <id>

    [PRIMARY]
    <sequence>

    [EVOLUTIONARY]
    (21 lines, each length L)

    [TERTIARY]
    (3 lines, each length 3*L) -> reshaped to (L,9)

    [MASK]
    <string of '+' and '-' OR '1' and '0'>

    Returns:
        List of dictionaries containing:
        id (str)
        primary (str)
        evolutionary (np.ndarray shape = 21 x L)
        tertiary (np.ndarray shape = L x 9)
        mask (np.ndarray shape = L)
    """

    proteins = []

    with open(filepath, "r") as f:
        line = f.readline()

        while line:
            tag = line.strip()

            if tag != "[ID]":
                line = f.readline()
                continue

            # -------- ID --------
            pid = f.readline().strip()

            # -------- PRIMARY --------
            if f.readline().strip() != "[PRIMARY]":
                raise ValueError(f"Expected [PRIMARY] after [ID] for {pid}")

            primary = f.readline().strip()
            L = len(primary)

            # -------- EVOLUTIONARY --------
            if f.readline().strip() != "[EVOLUTIONARY]":
                raise ValueError(f"Expected [EVOLUTIONARY] after [PRIMARY] for {pid}")

            evo_rows = []
            for _ in range(21):
                row = f.readline().strip()
                vals = [float(x) for x in row.split()]

                if len(vals) != L:
                    raise ValueError(
                        f"EVOLUTIONARY row length {len(vals)} != sequence length {L} for {pid}"
                    )

                evo_rows.append(vals)

            evolutionary = np.array(evo_rows, dtype=np.float32)

            # -------- TERTIARY --------
            if f.readline().strip() != "[TERTIARY]":
                raise ValueError(f"Expected [TERTIARY] after [EVOLUTIONARY] for {pid}")

            tert_axes = []

            for _ in range(3):
                row = f.readline().strip()
                vals = [float(x) for x in row.split()]

                if len(vals) != 3 * L:
                    raise ValueError(
                        f"TERTIARY axis length {len(vals)} != 3*L ({3*L}) for {pid}"
                    )

                tert_axes.append(vals)

            tert_arr = np.array(tert_axes, dtype=np.float32)  # (3, 3L)

            # reshape to (L, 9)
            tert_flat = np.ravel(tert_arr.T)
            tertiary = np.reshape(tert_flat, (L, 9))

            # -------- MASK --------
            if f.readline().strip() != "[MASK]":
                raise ValueError(f"Expected [MASK] after [TERTIARY] for {pid}")

            mask_line = f.readline().strip()

            if " " in mask_line:
                parts = mask_line.split()

                if len(parts) != L:
                    raise ValueError(f"MASK token count {len(parts)} != {L} for {pid}")

                mask = np.array(
                    [1 if p in ("+", "1") else 0 for p in parts],
                    dtype=np.int8
                )

            else:
                if len(mask_line) != L:
                    raise ValueError(f"MASK length {len(mask_line)} != {L} for {pid}")

                mask = np.fromiter(
                    (1 if c in ("+", "1") else 0 for c in mask_line),
                    dtype=np.int8,
                    count=L
                )

            proteins.append({
                "id": pid,
                "primary": primary,
                "evolutionary": evolutionary,
                "tertiary": tertiary,
                "mask": mask
            })

            line = f.readline()

    return proteins


# ---------------------------------------------------------------------


def proteins_to_table(input_path, max_proteins=None, sample_residues=None):
    """
    Convert parsed proteins into a residue-level dataframe.
    """

    proteins = parse_casp9(input_path)

    if max_proteins:
        proteins = proteins[:max_proteins]

    rows = []

    for protein in proteins:

        pid = protein["id"]
        seq = protein["primary"]
        evo = protein["evolutionary"]
        tert = protein["tertiary"]
        mask = protein["mask"]

        L = len(seq)

        residue_indices = np.arange(L)

        if sample_residues and sample_residues < L:
            residue_indices = np.random.choice(
                residue_indices,
                sample_residues,
                replace=False
            )

        for i in residue_indices:

            row = {
                "ProteinID": pid,
                "ResidueIndex": i + 1,
                "Residue": seq[i],
                "SequenceLength": L,
                "Mask": int(mask[i])
            }

            # Evolutionary features
            for j in range(21):
                row[f"Evo{j+1}"] = evo[j, i]

            # Extract C-alpha coordinates
            row["X"], row["Y"], row["Z"] = tert[i, 3:6]

            # flag valid coordinates
            row["ValidCoords"] = int(
                not (row["X"] == 0 and row["Y"] == 0 and row["Z"] == 0)
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    return df

