
import os
import warnings
import deepchem as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# 1. GAT-based GNN model
class GAT(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GAT, self).__init__()
        self.gat1 = GATConv(num_node_features, 64, heads=4, concat=True, dropout=0.2)
        self.gat2 = GATConv(64 * 4, 128, heads=1, concat=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(128, 1)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)  # Pooling per molecule
        return torch.sigmoid(self.linear(x)).view(-1)

# 2. SMILES → Graph conversion
def mol_to_graph(mol, label):
    from rdkit import Chem
    from rdkit.Chem import rdmolops

    if mol is None or mol.GetNumAtoms() == 0:
        return None

    node_feats = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    node_feats = torch.tensor(node_feats, dtype=torch.float)

    edges = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(edges).nonzero(), dtype=torch.long)

    return Data(x=node_feats, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))

# 3. Prepare data
def load_graph_data(task_name="SR-HSE"):
    print("Loading data...")
    tox21_tasks, datasets, _ = dc.molnet.load_tox21(featurizer="Raw", splitter="random")
    train_dataset, valid_dataset, _ = datasets
    task_index = tox21_tasks.index(task_name)

    from rdkit import Chem
    train_graphs, valid_graphs = [], []

    for dataset, container in [(train_dataset, train_graphs), (valid_dataset, valid_graphs)]:
        for i in range(len(dataset)):
            smi = dataset.ids[i]
            label = dataset.y[i][task_index]
            if np.isnan(label):
                continue
            mol = Chem.MolFromSmiles(smi)
            graph = mol_to_graph(mol, label)
            if graph is not None:
                container.append(graph)

    return train_graphs, valid_graphs

# 4. Training
def train():
    train_graphs, valid_graphs = load_graph_data()

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT(num_node_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    print("Training started...")
    model.train()
    for epoch in range(1, 21):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            y_trues.extend(batch.y.view(-1).cpu().numpy())
            y_preds.extend(out.cpu().numpy())

    auc = roc_auc_score(y_trues, y_preds)
    print(f"\nValidation AUC: {auc:.4f}")

    # Save results
    results_path = "/content/drive/MyDrive/drug_discovery_ml/results/sr_hse_gat.csv"
    pd.DataFrame({"y_true": y_trues, "y_pred": y_preds}).to_csv(results_path, index=False)
    print(f"Results saved: {results_path}")

if __name__ == "__main__":
    train()
