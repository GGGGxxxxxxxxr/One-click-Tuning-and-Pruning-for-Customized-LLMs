import torch
import torch.nn as nn
import torch.nn.functional as F


class moe_gate(nn.module):
    '''
    out_dim: num of experts waiting to be directed
    in_dim:  hidden_dim of the networks
    '''
    def __init__(self, in_dim, num_experts):
        super(moe_gate, self).__init__()
        self.fc   = nn.Linear(in_dim, num_experts)
        self.gate = F.softmax()
    
    def forward(self, input_features):
        temp            = self.fc(input_features)
        router_logits   = self.gate(temp)
        router_score    = F.softmax(router_logits)
        return router_score


class MLP(nn.Module):
    def __init__(self, in_dim, ffn_multiplier):
        super(MLP, self).__init__()
        self.intermediate_dim = in_dim * ffn_multiplier
        self.up_proj = nn.Linear(in_dim, self.intermediate_dim)
        self.gate_proj = nn.Linear(in_dim, self.intermediate_dim)
        self.down_proj = nn.Linear(self.intermediate_dim, in_dim)
        self.act_fn = nn.ReLU()
    
    def forward(self, input_features):
        temp = self.act_fn(self.gate_proj(input_features)) * self.up_proj(input_features)
        return self.down_proj(temp)


class moe_MLP(nn.Module):
    def __init__(self, in_dim, ffn_multiplier, num_experts):
        super(moe_MLP, self).__init__()
        self.experts = nn.ModuleList([MLP(in_dim=in_dim, ffn_multiplier=ffn_multiplier) for _ in range(num_experts)])
        self.moe_gate = moe_gate(in_dim=in_dim, out_dim=num_experts)
        
    def forward(self, in_features):
        routing_logits = F.softmax(self.moe_gate(in_features))



