import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.bias = nn.Parameter(torch.zeros(out_ft)) if bias else None

        # 权重初始化
        nn.init.xavier_uniform_(self.fc.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.spmm(adj, seq_fts)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out) if self.act is not None else out


class Model_Pretrain(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation):
        super(Model_Pretrain, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation
        self.gcn1 = GCN(in_dim, hidden_dim, activation)
        self.gcn2 = GCN(hidden_dim, hidden_dim, activation)

        self.proj_nc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.proj_ego = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.proj_nbr = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.fc_normal_prompt = nn.Linear(out_dim*2, out_dim*2, bias=False)
        self.fc_abnormal_prompt = nn.Linear(out_dim*2, out_dim*2, bias=False)
        self.prompt = SimplePrompt(out_dim * 2)
        self.act = nn.ReLU()

    def forward(self, feat, adj, ego_raw, nbr_raw, normal_prompt, abnormal_prompt, sparse=False):
        emb = self.gcn1(feat, adj, sparse=sparse)
        z = self.gcn2(emb, adj, sparse=sparse)
        z = self.proj_nc(z)
        h_ego, h_nbr = self.encode_attributes(ego_raw, nbr_raw)
        normal_prompt = normal_prompt.to(feat.device)
        normal_prompt = self.act(self.fc_normal_prompt(normal_prompt))
        enhanced_normal = self.prompt(normal_prompt)
        abnormal_prompt = abnormal_prompt.to(feat.device)
        abnormal_prompt = self.act(self.fc_abnormal_prompt(abnormal_prompt))
        enhanced_abnormal = self.prompt(abnormal_prompt)
        return h_ego, h_nbr, normal_prompt, abnormal_prompt, enhanced_normal, enhanced_abnormal, z

    def encode_attributes(self, x_ego, x_nbr):
        h_ego = self.proj_ego(x_ego)
        h_nbr = self.proj_nbr(x_nbr)
        return h_ego, h_nbr


class SimplePrompt(nn.Module):
    def __init__(self, input_size):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, input_size))
        self.a = nn.Linear(input_size, input_size)
        self.reset_parameters()
        self.act = nn.ReLU()

    def reset_parameters(self):
        glorot(self.global_emb)
        self.a.reset_parameters()

    def forward(self, x):
        return x + self.act(self.a(x)) + self.global_emb


# class SimplePrompt(nn.Module):
#     def __init__(self, input_size):
#         super(SimplePrompt, self).__init__()
#         self.global_emb = nn.Parameter(torch.Tensor(1, input_size))
#         self.psi = nn.Parameter(torch.Tensor(input_size))
#         self.scale = nn.Parameter(torch.tensor(0.1))
#         self.a = nn.Linear(input_size, input_size)
#         self.act = nn.ReLU()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.global_emb)  # [1, input_size] -> 2D，可用
#         nn.init.xavier_uniform_(self.psi.unsqueeze(0))  # 将 [D] 变成 [1, D] 再初始化
#         nn.init.normal_(self.a.weight, std=1e-3)
#         nn.init.zeros_(self.a.bias)
#
#     def forward(self, x):
#         base_enhance  = x + self.act(self.a(x)) + self.global_emb
#         proj_scalar = torch.dot(x, self.psi) / (x.norm()**2 + 1e-8)
#         proj_vector = proj_scalar * x
#         ortho_component = self.psi - proj_vector
#         ortho_component = F.normalize(ortho_component, p=2, dim=0)
#         final_enhanced = base_enhance + self.scale*ortho_component
#         return final_enhanced

# class GPFplusAtt(nn.Module):
#     def __init__(self, in_channels, p_num=600):
#         super(GPFplusAtt, self).__init__()
#         self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
#         self.a = nn.Linear(in_channels, p_num)
#         self.reset_parameters()
#         self.b = nn.Linear(in_channels, in_channels)
#         self.act = nn.ReLU()
#
#     def reset_parameters(self):
#         glorot(self.p_list)
#         self.a.reset_parameters()
#
#     def forward(self, x):
#         x = x.unsqueeze(0)
#         score = self.a(x)
#         weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
#         p = weight @ self.p_list
#         return x + p + self.act(self.b(x))
#
#
# import torch
# import torch.nn as nn
# from torch_geometric.nn.inits import glorot
# class ResidualPrompt(nn.Module):
#     def __init__(self, input_size):
#         super(ResidualPrompt, self).__init__()
#         self.input_size = input_size
#         self.global_emb = nn.Parameter(torch.Tensor(1, input_size))
#         self.trans = nn.Linear(input_size, input_size)
#         self.act = nn.ReLU()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         glorot(self.global_emb)
#         self.trans.reset_parameters()
#
#     def forward(self, x):
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         gamma = self.act(self.trans(self.global_emb))
#         enhanced = x * gamma
#         return x + enhanced