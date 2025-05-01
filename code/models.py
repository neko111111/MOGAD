import torch.nn as nn
import torch
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def mish(x):  # Mish激活函数
    return x * (torch.tanh(F.softplus(x)))


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = mish(torch.matmul(a_input, self.a).squeeze(2))  # [N,N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # print(attention)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return mish(h_prime)


class GAT(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout, nheads):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout)
        self.s = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return F.softmax(self.s(self.out_att(x, adj)), dim=1)


class NNFusion(nn.Module):
    def __init__(self, input_size,
                 hidden_size, num, output_size, dropout_rate=0):
        super(NNFusion, self).__init__()

        self.num = num

        self.fusion_linear = nn.Linear(input_size * num, hidden_size)

        self.output_linear = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_list):
        x = torch.cat([x for x in x_list], dim=1)

        fused_representation = F.relu(self.fusion_linear(x))

        fused_representation = self.dropout(fused_representation)

        output = F.softmax(self.output_linear(fused_representation), dim=1)

        return output


class AttentionFusion(nn.Module):
    def __init__(self, input_size,
                 hidden_size, num, output_size, dropout_rate=0):
        super(AttentionFusion, self).__init__()

        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) for i in range(num)])

        self.num = num

        # Define the attention weights
        self.attention_weights = nn.Parameter(torch.rand(num))

        # Linear transformation for the fused representation
        self.fusion_linear = nn.Linear(hidden_size * num, hidden_size)

        self.output_linear = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_list):

        for i, linear in enumerate(self.linear_layers):
            x_list[i] = F.relu(linear(x_list[i]))

        # Compute attention scores
        # attention_scores = F.softmax(self.attention_weights, dim=0)
        attention_scores = self.attention_weights

        # Apply attention to the inputs
        for i in range(self.num):
            x_list[i] = x_list[i] * attention_scores[i]

        # # 使用 torch.stack 将列表中的 tensors 堆叠起来
        # stacked_tensor = torch.stack(x_list)
        #
        # # 使用 torch.sum 将堆叠后的 tensors 相加
        # result_tensor = torch.sum(stacked_tensor, dim=0)

        result_tensor = torch.cat(x_list, dim=1)

        # Fuse the attention-weighted inputs
        fused_representation = F.relu(self.fusion_linear(result_tensor))

        # Apply dropout before the final classification layer
        fused_representation = self.dropout(fused_representation)

        # Apply the final classification layer with softmax activation
        output = F.softmax(self.output_linear(fused_representation), dim=1)

        return output


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, heads=1, dropout=0.6, alpha=0.2):
        super(GATConv, self).__init__()
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha

        # Define linear layers for each attention head
        self.W = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(heads)])
        self.a = nn.ModuleList([nn.Linear(2 * out_features, 1) for _ in range(heads)])

        # Dropout layer for regularization
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Initialize weights
        for i in range(heads):
            nn.init.xavier_uniform_(self.W[i].weight)
            nn.init.zeros_(self.W[i].bias)
            nn.init.xavier_uniform_(self.a[i].weight)
            nn.init.zeros_(self.a[i].bias)

    def forward(self, x, adj):
        head_outputs = []
        for i in range(self.heads):
            h = self.W[i](x)
            N = h.size(0)

            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * h.size(1))
            e = F.leaky_relu(self.a[i](a_input).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = self.dropout_layer(attention)

            h_prime = torch.matmul(attention, h)
            head_outputs.append(h_prime)

        # Concatenate the outputs of all attention heads
        output = torch.cat(head_outputs, dim=1)
        return output


class MultiGraphAttentionFusion(nn.Module):
    def __init__(self, num_graphs, input_dim, hidden_dim, output_dim, head=1, dropout=0.5):
        super(MultiGraphAttentionFusion, self).__init__()

        # Define GAT layers for each graph
        self.gat_layers = nn.ModuleList(
            [GATConv(input_dim[i], hidden_dim, heads=head, dropout=dropout) for i in range(num_graphs)])

        # Attention fusion weights
        self.attention_weights = nn.Parameter(torch.ones(num_graphs))

        # Final linear layer for fusion
        self.final_linear = nn.Linear(num_graphs * hidden_dim * head, output_dim)

    def forward(self, graphs):
        # graphs is a list of graph data (x, adj)

        # Apply GAT layers to each graph
        graph_outputs = []
        for i, graph_data in enumerate(graphs):
            x, adj = graph_data
            x = self.gat_layers[i](x, adj)
            graph_outputs.append(x)

        # Apply attention weights
        attention_weights = F.softmax(self.attention_weights, dim=0)
        fused_output = torch.cat([attention_weights[i] * graph_output for i, graph_output in enumerate(graph_outputs)],
                                 dim=1)

        # Final linear layer for fusion
        output = self.final_linear(fused_output)
        return F.softmax(output, dim=1)


def init_model_dict(num_view, num_class, dim_list, configs, mgaf=True):
    model_dict = {}
    if mgaf:
        num_af = num_view + 1
    else:
        num_af = num_view
    for i in range(num_view):
        model_dict["A{:}".format(i + 1)] = GAT(nfeat=dim_list[i], nclass=num_class, nhid=configs['hidden_mgat'],
                                               dropout=configs['drop_mgat'], nheads=configs['head_mgat'])

    model_dict['M'] = MultiGraphAttentionFusion(num_graphs=num_view, input_dim=dim_list,
                                                hidden_dim=configs['hidden_mgaf'],
                                                output_dim=num_class, dropout=configs['drop_mgaf'])

    model_dict["C"] = AttentionFusion(num_class, hidden_size=configs['hidden_af'], num=num_af,
                                      output_size=num_class)
    # model_dict["C"] = NNFusion(num_class, hidden_size=8, num=num_view+1, output_size=num_class)

    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4, lr_m=1e-3, weight_decay=3e-3, weight_decay_m=3e-3):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["A{:}".format(i + 1)] = torch.optim.Adam(
            model_dict["A{:}".format(i + 1)].parameters(),
            lr=lr_e, weight_decay=weight_decay)

    optim_dict["M"] = torch.optim.Adam(
        model_dict["M"].parameters(),
        lr=lr_m, weight_decay=weight_decay_m)
    optim_dict["C"] = torch.optim.Adam(
        model_dict["C"].parameters(),
        lr=lr_c)

    return optim_dict
