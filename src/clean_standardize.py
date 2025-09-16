# < is upper bound for pIC50
# salt-strip, neutralize, sanitize with RDKit
# deduplicate by canonicalized SMILES (aggregate duplicates with median nM per SMILES, type)
# convert to pActivity = 9 - log10(nM) (pX = -log10(M), and M = nM * 1e-9 > pX = 9 - log10(nM))
# cap to <1000 by stratified sampling across pActivity wuantiles

import pandas as pd
import numpy as np 
from rdkit import Chem
from rdkit.Chem import SaltRemover 

RAW = "data_raw/egfr_raw.csv"
OUT = "data_proc/egfr_clean.csv"
MAX_N = 1000

def to_mol(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None: return None
        Chem.SanitizeMol(m)
        return m 
    except:
        return None

def standardize_smiles(smiles):
    m = to_mol(smiles)
    if m is None: return None

    # salt strip using SaltRemover
    remover = SaltRemover.SaltRemover()
    m = remover.StripMol(m)

    # canonical SMILES
    return Chem.MolToSmiles(m, canonical=True)

def compute_pX(nM):
    # pX = 9 - log10(nM)
    return 9 - np.log10(nM)

def main():
    df = pd.read_csv(RAW)
    # prefer IC50 over Ki
    df["standard_type"] = df["standard_type"].fillna("IC50")
    df = df[df["standard_type"].isin(["IC50", "Ki"])].copy()

    #standardize SMILES
    df["std_smiles"] = df["canonical_smiles"].apply(standardize_smiles)
    df = df.dropna(subset=["std_smiles"])
    #remove obvious bad values
    df = df[(df["standard_value_nM"] > 0) & (df["standard_value_nM"] < 1e8)].copy()

    #choose one type per compound (prioritize IC50)
    #aggregate by std_smiles, standard_type -> median nM
    agg = (df.groupby(["std_smiles","standard_type"])["standard_value_nM"]
            .median()
            .reset_index())

    #keep ic50 rows, if SMILES is missing IC50, keeps Ki
    ic50 = agg[agg["standard_type"] == "IC50"]
    ki = agg[agg["standard_type"] == "Ki"]
    ic50_smiles = set(ic50["std_smiles"])
    ki = ki[~ki["std_smiles"].isin(ic50_smiles)]
    merged = pd.concat([ic50, ki], ignore_index=True)

    #pActivity
    merged["pActivity"] = merged["standard_value_nM"].apply(compute_pX)
    merged.rename(columns={"standard_type":"assay_type", "std_smiles":"smiles"}, inplace=True)

    #trim extremes and sample ≤1000 with stratification
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["pActivity"])
    merged = merged[(merged["pActivity"]>3) & (merged["pActivity"]<12)]
    
    #stratify by quantiles for a balanced distribution
    merged["bin"] = pd.qcut(merged["pActivity"], q=min(10, merged.shape[0]), duplicates="drop")
    if merged.shape[0] > MAX_N:
        merged = (merged.groupby("bin", group_keys=False)
                        .apply(lambda g: g.sample(min(MAX_N // merged["bin"].nunique(), len(g)), random_state=42)))
    merged = merged.drop(columns=["bin"])

    merged = merged[["smiles","assay_type","standard_value_nM","pActivity"]].reset_index(drop=True)
    merged.to_csv(OUT, index=False)
    print(merged.shape, "saved to", OUT)
    print(merged["assay_type"].value_counts())

if __name__ == "__main__":
    main()