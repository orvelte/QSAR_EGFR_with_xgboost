# fetch for CHEMBL203 (EGFR)
# EGFR entries with reliable IC50/Ki in nM + SMILES

import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGET_ID = "CHEMBL203" # EGFR
TYPES = {"IC50", "Ki"} # choose either later (pIC50 works for either if in nM)

def fetch_egfr():
    targets = new_client.target
    activities = new_client.activity
    target = targets.get(TARGET_ID)
    # single protein target only
    qs = {
        "target_chembl_id": TARGET_ID,
        "standard_type__in": list(TYPES),
        "standard_units": "nM",
        "limit": 5000,  # fetching extra then clean down to ~1000
    }
    recs = activities.filter(**qs)
    rows = []
    for r in recs:
        # keep only numeric, meaningful values with known relation to activity  
        if not r.get("canonical_smiles"):
            continue
        if r.get("standard_value") is None:
            continue
        try:
            val = float(r["standard_value"])
        except:
            continue
        rel = r.get("standard_relation") or "="
        if rel not in {"=", "<", "<="}:  # skip ">"
            continue
        rows.append({
            "molecule_chembl_id": r.get("molecule_chembl_id"),
            "canonical_smiles": r.get("canonical_smiles"),
            "standard_type": r.get("standard_type"),
            "standard_relation": rel,
            "standard_value_nM": val,
            "assay_chembl_id": r.get("assay_chembl_id"),
            "doc_chembl_id": r.get("document_chembl_id"),
        })
    df = pd.DataFrame(rows).dropna(subset=["canonical_smiles"])
    return df

if __name__ == "__main__":
    df = fetch_egfr()
    df.to_csv("data_raw/egfr_raw.csv", index=False)
    print(df.shape, "saved to data_raw/egfr_raw.csv")