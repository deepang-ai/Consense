import os
import sys

import copy
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch import nn, optim
from torch_geometric.data import DataLoader
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from ExploreFormer.model import GraphTransformer
from ExploreFormer.data import GraphDataset
from ExploreFormer.utils import count_parameters
from pyg_data import PygGraphDataset

def load_args():
    parser = argparse.ArgumentParser(description='Structure-Aware Transformer for vulnerability dection')

    parser.add_argument('--root-dir', type=str, default='./dataset/scvhunter/embeddings', help='name of datasets_dir')
    parser.add_argument('--dataset', type=str, default='reentrancy', help='name of dataset')

    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=3, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=512, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
    parser.add_argument('--gnn-type', type=str, default='gat', choices=['gat', 'gcn', 'gin'], help="GNN structure extractor type")
    
    parser.add_argument('--k-hop', type=int, default=2, help="number of layers for GNNs")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'add', 'gat', 'cls'], help='global pooling method')
    parser.add_argument('--se', type=str, default="khopgnn", help='Extractor type: khopgnn, or gnn')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')

    parser.add_argument('--seed', type=int, help='random seed', default=12)
    parser.add_argument('--outdir', type=str, default='./logs/train', help='output path')
    args = parser.parse_args()
    args.device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    args.save_logs = True
    if args.outdir != '':
        args.outdir = args.outdir + '/{}'.format(args.dataset)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = args.outdir + '/{}'.format(current_time)
        os.makedirs(args.outdir, exist_ok=True)
    return args


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    start_time = timer()
    for i, data in enumerate(loader):
        data_raw = data.to(device)
        optimizer.zero_grad()
        pred_out = model(data_raw)
        loss = criterion(pred_out, data_raw.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_raw.num_graphs
    epoch_loss = total_loss / len(loader.dataset)
    return epoch_loss, timer() - start_time


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_pred = []
    y_true = []
    start_time = timer()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            size = len(data.y)
            output = model(data)
            try:
                loss = criterion(output, data.y.squeeze())
            except:
                pass
            y_true.append(data.y.cpu())
            y_pred.append(output.argmax(dim=-1).view(-1, 1).cpu())
            running_loss += loss.item() * size
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    pre = precision_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    rec = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    score = (acc, pre, rec, f1)
    return score, epoch_loss, timer() - start_time


def train_one_fold(train_dataset, val_dataset, model, criterion, optimizer, args, fold):
    train_dset = GraphDataset(train_dataset, degree=True, k_hop=args.k_hop, se=args.se)
    val_dset = GraphDataset(val_dataset, degree=True, k_hop=args.k_hop, se=args.se)
    
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)


    print("Training...")
    best_val_score = (0, 0, 0, 0)
    best_weights = None
    best_epoch = 0
    start_time = timer()
    for epoch in range(args.epochs):
        train_loss, train_time = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print("Train\t {}/{}, loss {:.4f}, time {:.4f}"
              .format(epoch + 1, args.epochs, train_loss, train_time))
        
        val_score, val_loss, val_time = eval_epoch(model, val_loader, criterion, args.device)
        print("Val\t {}/{}, loss {:.4f}, time {:.4f} | acc {:.4f} | pre {:.4f} | recall {:.4f} | f1 {:.4f}"
              .format(epoch + 1, args.epochs, val_loss, val_time, val_score[0], val_score[1], val_score[2], val_score[3]))
        
        
        if val_score[0] > best_val_score[0]:
            best_val_score = val_score
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    
    total_time = timer() - start_time
    print("best epoch: {} acc: {:.4f} | Precision: {:.4f} | recall {:.4f} | f1 {:.4f}"
          .format(best_epoch, best_val_score[0], best_val_score[1], best_val_score[2], best_val_score[3]))
    print("Total time: {:.4f}".format(total_time))

    save_dir = args.outdir + '/fold_{}'.format(fold)
    os.makedirs(save_dir, exist_ok=True)

    if args.save_logs:
        torch.save(
            {'args': args,
                'state_dict': best_weights},
            save_dir + '/model.pth')

    return best_val_score



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = PygGraphDataset(name=args.dataset, root=args.root_dir)
    labels = [g.y.item() for g in dataset]


    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"Fold {fold + 1} / 5")

        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]

        input_size = dataset[0].x.size()[1]
        model = GraphTransformer(in_size=input_size,
                                num_class=dataset.num_classes,
                                d_model=args.dim_hidden,
                                dim_feedforward=int(0.5 * args.dim_hidden),
                                num_heads=args.num_heads,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                gnn_type=args.gnn_type,
                                k_hop=args.k_hop,
                                se=args.se,
                                global_pool=args.global_pool)
        print(model)
        model.to(args.device)
        print("Total number of parameters: {}".format(count_parameters(model)))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        flod_score = train_one_fold(train_dataset, val_dataset, model, criterion, optimizer, args, fold+1)
        fold_scores.append(flod_score)

    print("=================Final Scores: =================")
    score_dict = []

    score_dict.append({
        'metric': 'acc',
        'score': [i[0] for i in fold_scores],
        'mean': np.mean([i[0] for i in fold_scores]),
        'max': np.max([i[0] for i in fold_scores])
    })

    score_dict.append({
        'metric': 'prec',
        'score': [i[1] for i in fold_scores],
        'mean': np.mean([i[1] for i in fold_scores]),
        'max': np.max([i[1] for i in fold_scores])
    })

    score_dict.append({
        'metric': 'recall',
        'score': [i[2] for i in fold_scores],
        'mean': np.mean([i[2] for i in fold_scores]),
        'max': np.max([i[2] for i in fold_scores])
    })

    score_dict.append({
        'metric': 'f1',
        'score': [i[3] for i in fold_scores],
        'mean': np.mean([i[3] for i in fold_scores]),
        'max': np.max([i[3] for i in fold_scores])
    })

    df = pd.DataFrame(score_dict)
    df.to_csv(os.path.join(args.outdir, 'score.csv'), index=False, encoding='utf-8')
    print(f"Final mean score: acc: {score_dict[0]['mean']} | Precision: {score_dict[1]['mean']} | recall {score_dict[2]['mean']} | f1 {score_dict[3]['mean']}")

    
if __name__ == "__main__":
    global args
    args = load_args()
    sys.stdout = Logger(os.path.join(args.outdir, 'log.txt'), sys.stdout)
    print(args)
    main(args)