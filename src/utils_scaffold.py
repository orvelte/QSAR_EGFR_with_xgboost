from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def murcko_scaffold(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    core = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(core, canonical=True) if core is not None else None
    