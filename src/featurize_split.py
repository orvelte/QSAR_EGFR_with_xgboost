import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from utils_scaffold import murcko_scaffold

INP = "data_proc/egfr_clean.csv"
OUT_X = "data_proc/X.npy"
OUT_Y = "data_proc/y.npy"
OUT_META = "data_proc/meta.csv"

N_BITS = 2048
RADIUS = 2

def morgan_fp(smiles, n_bits=N_BITS, radius=RADIUS):
    m = Chem.MolFromSmiles(smiles)
    if m is None: 
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    df = pd.read_csv(INP)
    df["scaffold"] = df["smiles"].apply(murcko_scaffold)
    fps = []
    keep_idx = []
    for i, smi in enumerate(df["smiles"]):
        arr = morgan_fp(smi)
        if arr is not None:
            fps.append(arr)
            keep_idx.append(i)
    X = np.stack(fps)
    df = df.iloc[keep_idx].reset_index(drop=True)
    y = df["pActivity"].values

    # Scaffold split: put rare scaffolds into test first
    # Simple approach: group by scaffold, then split groups
    scaff_counts = df["scaffold"].value_counts()
    uniq_scaffs = list(scaff_counts.index)

    # 80/10/10 split at scaffold level
    rng = np.random.RandomState(42)
    rng.shuffle(uniq_scaffs)
    n = len(uniq_scaffs)
    n_train = int(0.8*n); n_val = int(0.1*n)
    train_scaffs = set(uniq_scaffs[:n_train])
    val_scaffs   = set(uniq_scaffs[n_train:n_train+n_val])
    test_scaffs  = set(uniq_scaffs[n_train+n_val:])

    def mask(scaff_set):
        return df["scaffold"].isin(scaff_set).values

    train_mask = mask(train_scaffs)
    val_mask   = mask(val_scaffs)
    test_mask  = mask(test_scaffs)

    meta = df[["smiles","assay_type","scaffold"]].copy()
    meta["split"] = np.where(train_mask, "train", np.where(val_mask, "val", "test"))

    # Save
    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    meta.to_csv(OUT_META, index=False)
    print("Shapes:", X.shape, y.shape, meta["split"].value_counts().to_dict())

if __name__ == "__main__":
    main()

