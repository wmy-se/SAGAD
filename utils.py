
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import dgl
import os


def preprocess_features(features):
    if sp.issparse(features):
        rowsum = np.array(features.sum(1)).flatten()
    else:
        rowsum = np.array(features.sum(1)).flatten()
    r_inv = np.power(rowsum, -1)
    r_inv[~np.isfinite(r_inv)] = 0.
    if sp.issparse(features):
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        features = features.todense()
    else:
        features = features * r_inv.reshape(-1, 1)
    return np.asarray(features)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def load_mat(dataset, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_train = sio.loadmat("./data/{}.mat".format(dataset))
    label_train = data_train['Label'] if ('Label' in data_train) else data_train['gnd']
    attr_train = data_train['Attributes'] if ('Attributes' in data_train) else data_train['X']
    network_train = data_train['Network'] if ('Network' in data_train) else data_train['A']
    if sp.issparse(attr_train):
        attr_train = attr_train.todense()
    X = torch.FloatTensor(np.array(attr_train))
    if sp.issparse(network_train):
        A = network_train.tocsr()
    else:
        A = sp.csr_matrix(network_train)
    y = label_train.astype(int).flatten()
    assert len(y) == X.shape[0] == A.shape[0], "数据维度不匹配!"
    print(f"数据集 {dataset} 加载完成！")
    print(f"  总节点数: {len(y)} | 特征维度: {X.shape[1]}")
    print(f"  正常节点: {np.sum(y == 0)} | 异常节点: {np.sum(y == 1)}")
    return X, A, y

def aggregation(graph, feat, k):
    with graph.local_scope():
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).unsqueeze(1).to(feat.device)
        h = feat
        for _ in range(k):
            h = h * norm
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = graph.ndata.pop('h')
            h = h * norm
        return h


def get_diag(graph, k):
    eye = torch.eye(graph.num_nodes(), device=graph.device)
    agg_mat = aggregation(graph, eye, k)
    return torch.diag(agg_mat)


def SND(feat, adj, k=1, device='cuda', datasetname=None, cache_dir="./cache"):
    assert datasetname is not None, "please provide 'datasetname' for caching"
    from pathlib import Path
    import pickle
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    cache_file = cache_path / f"{datasetname}_{k}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            ego_feats, nbr_feats = pickle.load(f)
        return ego_feats.to(device), nbr_feats.to(device)

    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            adj = adj.coalesce()
            row, col = adj.indices()[0], adj.indices()[1]
        else:
            row, col = adj.nonzero(as_tuple=True)
    else:
        adj_dense = torch.tensor(adj) if not torch.is_tensor(adj) else adj
        row, col = adj_dense.nonzero(as_tuple=True)
    src = row.long().cpu()
    dst = col.long().cpu()
    g = dgl.graph((src, dst), num_nodes=adj.shape[0])
    g = g.to(device)
    g = g.add_self_loop()
    if isinstance(feat, np.ndarray):
        feat_tensor = torch.FloatTensor(feat).to(device)
    else:
        feat_tensor = feat.to(device)
    aggregated = aggregation(g, feat_tensor, k)
    weight_diag = get_diag(g, k)
    self_contrib = (feat_tensor.T * weight_diag).T
    neighbor_feats = aggregated - self_contrib
    ego_feats=feat_tensor
    nbr_feats=neighbor_feats
    with open(cache_file, 'wb') as f:
        pickle.dump((ego_feats.cpu().detach(), nbr_feats.cpu().detach()), f)
    print(f"[PAS] Saved cached result for '{datasetname}' to {cache_file}")
    return ego_feats, nbr_feats


def get_community_by_metis(z, adj, labels, c_num=5, datasetname=None):
    assert datasetname is not None, "datasetname must be provided"
    device = z.device
    N = z.size(0)
    k = min(c_num, N)
    os.makedirs('./saved_idx', exist_ok=True)
    cluster_id_path = f'./saved_idx/{datasetname}.npy'
    if os.path.exists(cluster_id_path):
        cluster_id_np = np.load(cluster_id_path)
        cluster_id = torch.tensor(cluster_id_np, dtype=torch.long, device=device)
    else:
        print(f"[INFO] Running METIS partitioning for {datasetname}...")
        if isinstance(adj, torch.Tensor):
            if adj.is_sparse:
                adj = adj.coalesce()
                row, col = adj.indices()[0], adj.indices()[1]
            else:
                row, col = adj.nonzero(as_tuple=True)
        else:
            adj = torch.tensor(adj) if not torch.is_tensor(adj) else adj
            row, col = adj.nonzero(as_tuple=True)

        src = row.long().cpu()
        dst = col.long().cpu()

        g = dgl.graph((src, dst), num_nodes=N)
        g = dgl.to_simple(g)
        g = g.to('cpu')
        g = g.add_self_loop()
        from dgl import metis_partition_assignment
        cluster_id_cpu = metis_partition_assignment(g, k=k)
        cluster_id = cluster_id_cpu.to(device)
        np.save(cluster_id_path, cluster_id_cpu.numpy())
        print(f"[INFO] Saved cluster_id to {cluster_id_path}")

    community_center = torch.zeros(k, z.size(1), device=device)
    counts = torch.zeros(k, device=device)
    normal_mask = (labels == 0)
    with torch.no_grad():
        for i in range(N):
            if normal_mask[i]:
                cid = cluster_id[i].item()
                community_center[cid] += z[i]
                counts[cid] += 1
        counts = counts.clamp(min=1)
        community_center = community_center / counts.unsqueeze(1)
    z_community = community_center[cluster_id]
    nc_residual = z - z_community
    return z_community, nc_residual


def normalize_metric(x):
    if x.min() == x.max():
        return torch.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

#
# def construct_contrastive_loss(nc_residual, labels):
#     nc_residual = torch.norm(nc_residual, p=2, dim=1)  # [N]
#     nc_residual = normalize_metric(nc_residual)
#     loss = F.binary_cross_entropy(nc_residual, labels.float())
#     return loss


def prototype_contrastive_loss(sa_residual, normal_prompt, abnormal_prompt, labels, margin=1.0):
    device = sa_residual.device
    N, d = sa_residual.shape
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device)
    r = F.normalize(sa_residual, p=2, dim=1)
    p_n = F.normalize(normal_prompt.squeeze(), p=2, dim=0)
    p_a = F.normalize(abnormal_prompt.squeeze(), p=2, dim=0)
    sim_r_pn = torch.matmul(r, p_n)
    sim_r_pa = torch.matmul(r, p_a)
    target_sim = torch.where(labels == 0, sim_r_pn, sim_r_pa)
    opposite_sim = torch.where(labels == 1, sim_r_pn, sim_r_pa)
    loss_align_pos = -torch.mean(target_sim)
    loss_align_neg = torch.mean(F.relu(opposite_sim + margin))
    loss_align = loss_align_pos + loss_align_neg
    return loss_align


def compute_anomaly_scores_pretrain(sa_res, enhanced_normal, enhanced_abnormal, beta=1.0):
    device = enhanced_normal.device
    sa_res = sa_res.to(device)
    p_n = F.normalize(enhanced_normal.squeeze(), p=2, dim=0)
    p_a = F.normalize(enhanced_abnormal.squeeze(), p=2, dim=0)
    r = F.normalize(sa_res, p=2, dim=1)
    sim_to_normal = torch.sum(r * p_n.unsqueeze(0), dim=1)
    sim_to_abnormal = torch.sum(r * p_a.unsqueeze(0), dim=1)
    sa_score_a = torch.sigmoid(sim_to_abnormal)
    sa_score_n = torch.sigmoid(-sim_to_normal)
    sa_score = sa_score_a + beta * sa_score_n
    sa_score = normalize_metric(sa_score)
    sa_score = torch.nan_to_num(sa_score, nan=0.0, posinf=1.0, neginf=0.0)
    anomaly_scores = sa_score
    return anomaly_scores