class PESLModule(nn.Module):
    """ PESL 全局原型增强模块 (hidden_size = 768) """

    def __init__(self, hidden_dim):
        super(PESLModule, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_local):
        cls_embeddings = h_local[:, 0, :]  # [batch_size, 768]
        p_d = torch.mean(cls_embeddings, dim=0, keepdim=True)  # [1, 768]
        p_d_expanded = p_d.unsqueeze(1).expand_as(h_local)  # [batch_size, seq_len, 768]
        h_sem = self.layer_norm(h_local + p_d_expanded)
        return h_sem


class GCNLayer(nn.Module):
    """ GCN 句法依存卷积层 (输入/输出维度匹配 hidden_size = 768) """

    def __init__(self, hidden_dim):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, h, norm_adj):
        support = torch.matmul(h, self.weight)  # [batch_size, seq_len, 768]
        output = torch.matmul(norm_adj, support) + self.bias
        return torch.relu(output)


class PESL_GCN_Model(nn.Module):
    """ PESL-GCN 完整双流网络 """

    def __init__(self, config):
        super(PESL_GCN_Model, self).__init__()

        # 1. 自动加载预训练模型的 config.json 配置
        self.bert_config = AutoConfig.from_pretrained(config.pretrained_model_path)

        # 2. 载入预训练 SciBERT 骨干网络
        self.bert = AutoModel.from_pretrained(
            config.pretrained_model_path,
            config=self.bert_config
        )

        # 获取 SciBERT 的 hidden_size (768)
        embed_dim = self.bert_config.hidden_size

        # 3. 初始化双流模块
        self.pesl_stream = PESLModule(embed_dim)
        self.gcn_stream = nn.ModuleList([
            GCNLayer(embed_dim) for _ in range(config.gcn_num_layers)
        ])

        # 4. 分类器与 Dropout
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(embed_dim, config.num_classes)

    def forward(self, input_ids, attention_mask, adj_matrix):
        # SciBERT 编码层输出 -> [batch_size, seq_len, 768]
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_local = bert_outputs.last_hidden_state

        # 语义增强流 (PESL Stream)
        h_sem = self.pesl_stream(h_local)

        # 句法图卷积流 (GCN Stream)
        h_struct = h_local
        for gcn_layer in self.gcn_stream:
            h_struct = gcn_layer(h_struct, adj_matrix)

        # 残差融合: H_final = H_sem + H_struct
        h_final = h_sem + h_struct

        # 分类预测
        h_final = self.dropout(h_final)
        logits = self.classifier(h_final)  # [batch_size, seq_len, num_classes]

        return logits