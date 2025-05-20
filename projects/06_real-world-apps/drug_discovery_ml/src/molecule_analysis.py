
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np

def extract_descriptors(smiles_list):
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    descriptors = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc_values = []
            for name in descriptor_names:
                try:
                    desc_func = getattr(Descriptors, name)
                    desc_values.append(desc_func(mol))
                except:
                    desc_values.append(np.nan)
            descriptors.append(desc_values)
        else:
            descriptors.append([np.nan] * len(descriptor_names))

    df = pd.DataFrame(descriptors, columns=descriptor_names)
    return df

def main():
    tox21_tasks, datasets, _ = dc.molnet.load_tox21(featurizer='ECFP')
    train_dataset, _, _ = datasets
    smiles = train_dataset.ids

    print(f"Extracting descriptors from {len(smiles)} molecules...")
    df = extract_descriptors(smiles)
    df["smiles"] = smiles

    output_path = "/content/drive/MyDrive/drug_discovery_ml/results/tox21_descriptors.csv"
    df.to_csv(output_path, index=False)
    print(f"Descriptor dataset saved to {output_path}")

if __name__ == "__main__":
    main()
