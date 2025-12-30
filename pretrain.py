from utils import *
from model import *
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import random
import dgl
import argparse
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def pretrain(model, dataset, feat, adj, labels, split, c_num=5, num_epoch=300, lr=1e-4, weight_decay=1.0, beta=1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_true = torch.LongTensor(labels).to(device)

    best_val_auc = 0.0
    best_test_auc_roc = 0.0
    best_test_auc_pr = 0.0
    best_epoch = 0
    auc_roc_list = []
    auc_pr_list = []
    idx_train = split['train']
    idx_val = split['val']

    for epoch in range(num_epoch):
        model.train()
        optimizer.zero_grad()
        normal_prompt_raw = nn.Parameter(torch.randn(args.out_dim*2, device=device, requires_grad=True))
        abnormal_prompt_raw = nn.Parameter(torch.randn(args.out_dim*2, device=device, requires_grad=True))
        ego_raw, nbr_raw = SND(feat, adj, k=2, device=device, datasetname=dataset)
        h_ego, h_nbr, normal_prompt, abnormal_prompt, _, _, z = model(
            feat.to(device),
            adj.to(device),
            ego_raw.to(device),
            nbr_raw.to(device),
            normal_prompt_raw.to(device),
            abnormal_prompt_raw.to(device)
        )
        pas_residual = h_ego - h_nbr
        pas_residual = F.normalize(pas_residual, p=2, dim=1)
        with torch.no_grad():
            z_community, nc_residual = get_community_by_metis(
                z.detach(), adj, labels, c_num=c_num, datasetname=dataset
            )
        nc_residual = F.normalize(nc_residual, p=2, dim=1)  # [N, d]
        sa_residual = torch.cat([pas_residual,nc_residual], dim=1) # [N, 2*d]

        loss_align = prototype_contrastive_loss(sa_residual[idx_train], normal_prompt, abnormal_prompt, labels[idx_train])
        total_loss = loss_align
        total_loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == num_epoch - 1:
            model.eval()
            with torch.no_grad():
                scores = compute_anomaly_scores_pretrain(
                    sa_res=sa_residual,
                    enhanced_normal=normal_prompt,
                    enhanced_abnormal=abnormal_prompt,
                    beta=beta
                )
            scores_np = scores.cpu().numpy()
            labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            auc_roc = roc_auc_score(labels_np[idx_val], scores_np[idx_val])
            auc_pr = average_precision_score(labels_np[idx_val], scores_np[idx_val],
                                             average='macro',
                                             pos_label=1,
                                             sample_weight=None)
            auc_roc_list.append(auc_roc)
            auc_pr_list.append(auc_pr)
            if auc_roc > best_val_auc:
                best_val_auc = auc_roc
                best_test_auc_roc = auc_roc
                best_test_auc_pr = auc_pr
                best_epoch = epoch
                save_path = f"./pth/model_weights_{args.dataset_train}_{args.num_epoch}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved: {save_path} (AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f})")
            print(f"Epoch {epoch:3d}, Loss: {total_loss.item():.4f}, "
                  f"Align Loss: {loss_align.item():.4f}")
            print(f"            AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")

    print("\n" + "=" * 60)
    print("\n最佳模型性能（基于 Val AUC 选择）:")
    print(f"   Epoch: {best_epoch}")
    print(f"   → Test AUC-ROC: {best_test_auc_roc:.4f}")
    print(f"   → Test AUC-PR:  {best_test_auc_pr:.4f}")
    print(f"\n对比：Best Epoch 的 Val AUC-ROC: {best_val_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_train', type=str,default='Facebook_svd')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=300)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--c_num', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    print('Dataset Train: ', args.dataset_train)
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, A, y= load_mat(args.dataset_train)
    np.random.seed(args.seed)
    random.seed(args.seed)
    idx_all = np.arange(len(y))
    idx_train, idx_val, _, _ = train_test_split(
        idx_all, y,
        train_size=0.7,
        stratify=y,
        random_state=args.seed,
        shuffle=True
    )
    idx_train = idx_train.tolist()
    idx_val = idx_val.tolist()
    split={
        'train': idx_train,
        'val': idx_val
    }

    feat = preprocess_features(X)
    feat = torch.FloatTensor(feat).to(device)
    adj_sparse = A + A.T
    adj_sparse = (adj_sparse > 0).astype(int)
    adj_sparse.setdiag(0)
    adj_sparse.eliminate_zeros()
    adj_normalized = normalize_adj(adj_sparse)
    adj_with_self_loop = adj_normalized + sp.eye(adj_normalized.shape[0])
    adj_with_self_loop = adj_with_self_loop.tocoo()
    adj_with_self_loop = adj_with_self_loop.astype(np.float32)
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(int))  # [2, E]
        values = torch.from_numpy(sparse_mx.data)  # [E,]
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(adj_with_self_loop)
    labels = y.flatten().astype(int)

    N = feat.shape[0]
    D = feat.shape[1]
    model = Model_Pretrain(in_dim=D, hidden_dim=args.hidden_dim, out_dim=args.out_dim, activation='prelu')

    print(f"\n开始预训练 SAGAD（使用 {args.dataset_train} 数据）...")
    pretrain(
        model,
        args.dataset_train,
        feat,
        adj,
        labels,
        split,
        args.c_num,
        args.num_epoch,
        args.lr,
        args.weight_decay,
        args.beta
    )
    print("\n预训练完成！")