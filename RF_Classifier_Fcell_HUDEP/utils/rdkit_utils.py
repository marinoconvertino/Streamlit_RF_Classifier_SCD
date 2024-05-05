from rdkit.Chem import PandasTools, AllChem, DataStructs, MolToSmiles
import numpy as np
import pandas as pd

def add_molecule_column(df, smiles_col='SMILES', mol_col='Molecule'):
    PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol=smiles_col, molCol=mol_col)
    #df['SMILES'] = df['Molecule'].apply(mol_to_smiles)

def compute_morgan_fp(mol, depth=2, nBits=2048):
    a = np.zeros(nBits)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, depth, nBits), a)
    return a

def generate_morgan_fps(df, smiles_col='SMILES', mol_col='Molecule', fp_col='Morgan2FP'):
    df[fp_col] = df[mol_col].map(compute_morgan_fp)
    # -
    fps = df.Morgan2FP.to_list()
    df2 = pd.DataFrame(np.vstack(fps)) # takes the descriptor array and expands it Pandas columns
    df3 = pd.concat([df, df2], axis=1)
    #df3.iloc[0].to_list()
    df3.drop(columns=['Molecule'], inplace=True)
    df3.columns = df3.columns.astype(str)
    return df3