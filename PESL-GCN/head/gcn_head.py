import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    """
    单层句法依存图卷积单元 (Eq. 6)
    包含线性映射 W, 偏置 b, 邻接矩阵乘法与 Dropout 激活
    """

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.5):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 可学习参数矩阵 W \in \mathbb{R}^{d \times d} 与偏置 b
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier 均匀初始化参数矩阵"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算:
        :param h: 输入特征矩阵 [Batch_Size, Seq_Len, Hidden_Dim]
        :param adj: 句法依存邻接矩阵 [Batch_Size, Seq_Len, Seq_Len]
        :return: 图卷积后的句法特征表示 [Batch_Size, Seq_Len, Hidden_Dim]
        """
        # 1. 线性投影: H * W -> [Batch, Seq_Len, Out_Features]
        support = torch.matmul(h, self.weight)

        # 2. 拓扑邻接矩阵聚合: A * (H * W) + b
        output = torch.matmul(adj, support) + self.bias

        # 3. 激活函数与 Dropout 归一化
        output = F.relu(output)
        return self.dropout(output)


class GraphConvolutionHead(nn.Module):
    """
    多层句法依存图卷积网络 Head (Eq. 6)
    支持自动计算 D^{-1/2} A D^{-1/2} 自适应度矩阵归一化与跨层残差连接
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.5):
        super(GraphConvolutionHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 构建多层 GCN 堆叠
        self.gcn_layers = nn.ModuleList([
            GraphConvolutionLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 残差连接归一化层
        self.layer_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def normalize_adjacency(adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        度矩阵自适应对称归一化: \tilde{A} = D^{-1/2} A D^{-1/2}
        保证在图卷积计算时梯度不会爆炸或消失
        """
        batch_size, seq_len, _ = adj_matrix.shape
        # 计算节点的度 (Degree)
        degree = torch.sum(adj_matrix, dim=-1)  # [Batch, Seq_Len]

        # 防止 0 除异常 (Degree < 1e-12 时平滑处理)
        degree_inv_sqrt = torch.pow(torch.clamp(degree, min=1e-12), -0.5)

        # 构建对角度矩阵 D^{-1/2}
        degree_matrix = torch.diag_embed(degree_inv_sqrt)  # [Batch, Seq_Len, Seq_Len]

        # \tilde{A} = D^{-1/2} * A * D^{-1/2}
        norm_adj = torch.matmul(torch.matmul(degree_matrix, adj_matrix), degree_matrix)
        return norm_adj

    def forward(self, h_local: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        :param h_local: SciBERT 提取的局部语义特征 [Batch, Seq_Len, 1024]
        :param adj_matrix: 句法依存树构成的邻接矩阵 [Batch, Seq_Len, Seq_Len]
        """
        # 1. 邻接矩阵自适应拉普拉斯归一化
        norm_adj = self.normalize_adjacency(adj_matrix)

        # 2. 多层图卷积堆叠与句法特征拓扑演化
        h_struct = h_local
        for gcn_layer in self.gcn_layers:
            h_struct = gcn_layer(h_struct, norm_adj)

        # 3. 残差相加并应用 Layer Normalization
        h_struct_final = self.layer_norm(h_struct + h_local)
        return h_struct_final