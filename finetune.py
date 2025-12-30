import torch
import torch.nn as nn
from model import *
from utils import *
from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def finetune(model, feat, adj, labels, dataset, split, num_epoch=200, lr=1e-5, c_num=5, beta=1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam([
        {'params': list(filter(lambda p: p.requires_grad, model_pretrain.parameters()))}
    ], lr=lr)

    best_val_auc = 0.0
    best_val_auc_roc = 0.0
    best_val_auc_pr = 0.0
    best_epoch = 0
    auc_roc_list = []
    auc_pr_list = []
    train_idx = split['train']
    val_idx = split['val']

    for epoch in range(num_epoch):
        model_pretrain.train()
        optimizer.zero_grad()
        normal_prompt_raw = nn.Parameter(torch.randn(args.out_dim*2, device=device, requires_grad=False))
        abnormal_prompt_raw = nn.Parameter(torch.randn(args.out_dim*2, device=device, requires_grad=False))
        ego_raw, nbr_raw = SND(feat, adj, k=1, device=device, datasetname=dataset)
        h_ego, h_nbr, _, _, enhanced_normal, enhanced_abnormal, z = model(
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
                z.detach(), adj, labels, c_num=c_num, datasetname=args.dataset_test
            )
        nc_residual = F.normalize(nc_residual, p=2, dim=1)
        sa_residual = torch.cat([pas_residual, nc_residual], dim=1)
        # sa_residual = model.tune(sa_residual)
        # sa_residual = torch.matmul(sa_residual,W)

        fewshot_sa_residual = sa_residual[train_idx]
        distances = torch.norm(fewshot_sa_residual - enhanced_normal, dim=1)
        loss_one_cla = distances.mean()
        total_loss = loss_one_cla

        total_loss.backward()
        optimizer.step()
        if epoch % 2 == 0 and epoch != 0:
            print("<<<<<<<<<<<<<<<<<<<<<< Evaluation Begin>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            model_pretrain.eval()
            with torch.no_grad():
                scores = compute_anomaly_scores_pretrain(
                    sa_res = sa_residual,
                    enhanced_normal = enhanced_normal,
                    enhanced_abnormal = enhanced_abnormal,
                    beta=beta
                )
                scores_np = scores.cpu().numpy()
                labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
                auc_roc_val = roc_auc_score(labels_np[val_idx], scores_np[val_idx])
                auc_pr_val = average_precision_score(
                                                        labels_np[val_idx], scores_np[val_idx],
                                                        average='macro',
                                                        pos_label=1,
                                                        sample_weight=None
                                                     )
            if auc_roc_val > best_val_auc:
                best_val_auc = auc_roc_val
                best_val_auc_roc = auc_roc_val
                best_val_auc_pr = auc_pr_val
                best_epoch = epoch
            auc_roc_list.append(auc_roc_val)
            auc_pr_list.append(auc_pr_val)

            print("Epoch:", '%04d' % (epoch), "loss =", "{:.5f}".format(total_loss.item()))
            print(f"Val AUC-ROC: {auc_roc_val:.4f}, AUC-PR: {auc_pr_val:.4f}")
            print("<<<<<<<<<<<<<<<<<<<<<< Evaluation End >>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("\n" + "=" * 60)
    print(f"   Best model updated at epoch {best_epoch}:")
    print(f"   Val AUC-ROC: {best_val_auc_roc:.4f}, AUC-PR: {best_val_auc_pr:.4f}")


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_test', type=str, default='reddit_svd')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--num_few_shot', type=int, default=1)
parser.add_argument('--c_num', type=int, default=5)
parser.add_argument('--beta', type=float, default=0.9)
args = parser.parse_args()

dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, A, y = load_mat(args.dataset_test)
feat = preprocess_features(X)
feat = torch.FloatTensor(feat).to(device)
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
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(int))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).to(device)

adj = sparse_mx_to_torch_sparse_tensor(adj_with_self_loop)
labels = y.flatten().astype(int)
num_nodes = feat.shape[0]
idx_normal_all = np.where(labels == 0)[0]
idx_abnormal_all = np.where(labels == 1)[0]
assert len(idx_normal_all) > 0, "No normal nodes found in the dataset."
assert len(idx_abnormal_all) > 0, "No abnormal nodes found in the dataset."

rng = np.random.default_rng(seed=args.seed)
np.random.shuffle(idx_normal_all)
np.random.shuffle(idx_abnormal_all)
assert args.num_few_shot <= len(idx_normal_all), (f"Not enough normal nodes for few-shot training. "
                                                  f"Required: {args.num_few_shot}, Available: {len(idx_normal_all)}")
idx_normal_some = idx_normal_all[:100]
idx_train_few = idx_normal_some[:args.num_few_shot]
idx_val_remaining = np.setdiff1d(np.arange(num_nodes), idx_train_few)
split = {
    'train': idx_train_few,
    'val': idx_val_remaining,
    'normal_all': idx_normal_all,
    'abnormal_all': idx_abnormal_all
}

N = feat.shape[0]
D = feat.shape[1]
model_pretrain = Model_Pretrain(in_dim=D, hidden_dim=args.hidden_dim, out_dim=args.out_dim, activation='prelu')
model_pretrain.to(device)
checkpoint = torch.load('pth/model_weights_Facebook_svd_300.pth')
model_pretrain.load_state_dict(checkpoint, strict=False)

for param in model_pretrain.parameters():
    param.requires_grad = False
model_pretrain.prompt.a.weight.requires_grad = True
model_pretrain.prompt.a.bias.requires_grad = True
model_pretrain.prompt.global_emb.requires_grad = True

print("\n>>> Trainable parameters:")
for name, param in model_pretrain.named_parameters():
     print(f"{name}: requires_grad = {param.requires_grad}")

print(f"\n开始少样本再训练 MyGAD（使用 {args.dataset_test} 数据）...")
finetune(
    model=model_pretrain,
    feat=feat,
    adj=adj,
    labels=labels,
    dataset=args.dataset_test,
    split=split,
    num_epoch=args.num_epoch,
    lr=args.lr,
    c_num=args.c_num,
    beta=args.beta
)
print(f"\n少样本再训练完成! 使用正常节点的个数为{args.num_few_shot}, 测试数据集为{args.dataset_test}")