import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)  # Uniform distribution sampling
    return -torch.log(-torch.log(U + eps) + eps)  # Gumbel sampling formula

def hard_sample(out):
    binary_out = torch.round(out)  
    binary_out = (binary_out - out).detach() + out  
    return binary_out

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)  # Round number to nearest multiple

def gumbel_sigmoid_sample(logits, T, offset=0):  # Gumbel-Softmax sampling
    gumbel_sample = sample_gumbel(logits.size())
    gumbel_sample = gumbel_sample.to(logits.device)  # Match device
    y = logits + gumbel_sample + offset
    return F.sigmoid(y / T)

class hypernetwork(nn.Module):
    def __init__(self, t_structures, group_size=1, hidden_size=32):
        super(hypernetwork, self).__init__()
        self.T = 0.4  # Temperature for Gumbel-Sigmoid
        self.base = 3.0  # Offset
        self.group_size = group_size  
        self.h0 = torch.zeros(2, 1, int(2 * hidden_size))  # Initial GRU hidden state
        self.ln_tp = nn.LayerNorm([int(4 * hidden_size)])
        self.bi_GRU = nn.GRU(hidden_size, int(2 * hidden_size), bidirectional=True)
        self.t_sp = t_structures
        self.linear_list_tp = nn.ModuleList(
            [nn.Linear(int(4 * hidden_size), int(self.t_sp[i]), bias=False) for i in range(len(self.t_sp))]
        )
        self.inputs = nn.Parameter(torch.Tensor(len(t_structures), 1, hidden_size))
        nn.init.normal_(self.inputs)
        self.inputs.requires_grad = False

    def forward(self):
        self.h0 = self.h0.to(self.ln_tp.weight.device)
        outputs, _ = self.bi_GRU(self.inputs, self.h0)
        tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]
        tp_out = [self.linear_list_tp[i](tp_out[i]) for i in range(len(self.linear_list_tp))]
        
        if not self.training:  # Inference: Convert soft outputs to hard masks
            tp_out = [gumbel_sigmoid_sample(tp_out[i], offset=self.base, T=self.T).squeeze() for i in range(len(self.linear_list_tp))]
            soft_tp_out = tp_out
            tp_out = [hard_sample(tp_out[i]) for i in range(len(self.linear_list_tp))]
            for i in range(len(tp_out)):  # Ensure non-zero masks
                if tp_out[i].sum() == 0:
                    tp_out[i][soft_tp_out[i].argmax()] = 1
        else:  # Training: Generate soft pruning scores
            tp_out = [gumbel_sigmoid_sample(tp_out[i], offset=self.base, T=self.T).squeeze() for i in range(len(self.linear_list_tp))]

        return tp_out
    
    def hard_output(self):  # Generate hard masks
        self.h0 = self.h0.to(self.ln_tp.weight.device)
        outputs, _ = self.bi_GRU(self.inputs, self.h0)
        tp_out = [F.gelu(self.ln_tp(outputs[i, :])) for i in range(len(self.linear_list_tp))]
        tp_out = [self.linear_list_tp[i](tp_out[i]) for i in range(len(self.linear_list_tp))]
        tp_out = [gumbel_sigmoid_sample(tp_out[i], offset=self.base, T=self.T).squeeze() for i in range(len(self.linear_list_tp))]
        tp_out = [hard_sample(tp_out[i]) for i in range(len(self.linear_list_tp))]
        return tp_out

class virtual_basic_operation(nn.Module):
    def __init__(self, dim, ex_dict={}):
        super().__init__()
        self.dim = dim
        self.pruning_vector = torch.ones(dim)  # Initialize pruning vector (all ones)
        self.ex_dict = ex_dict  
    
    def forward(self, input):
        if len(input.size()) == 4:  # 4D: Convolutional feature maps
            p_v = self.pruning_vector[None, None, None, :]
        elif len(input.size()) == 3:  # 3D: Sequence data
            p_v = self.pruning_vector[None, None, :]
        elif len(input.size()) == 2:  # 2D: Fully connected layers
            p_v = self.pruning_vector[None, :]
        p_v = p_v.to(input.device)  # Match device
        return p_v.expand_as(input) * input  # Element-wise multiplication
    
    def set_vector_value(self, value):
        assert value.squeeze().size() == self.pruning_vector.squeeze().size()
        self.pruning_vector = value.squeeze() if value is not None else value

    def get_parameters(self):
        return 0

class virtual_block_basic_operation(virtual_basic_operation):
    def __init__(self, dim, ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)

class virtual_att_operation(virtual_basic_operation):
    def __init__(self, dim, ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)
        self.head_dim = ex_dict['head_dim']
        
    def get_parameters(self):
        return self.ex_dict['dim_1'] * self.ex_dict['dim_2'] * self.ex_dict['num_weight']
    
    def forward(self, input):
        if len(input.size()) == 4:  # Attention tensors
            p_v = self.pruning_vector[None, None, :, None]
            p_v = p_v.to(input.device)
            return p_v.expand_as(input) * input

class virtual_block_attn_operation(virtual_basic_operation):
    def __init__(self, dim, ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)
        self.head_dim = ex_dict['head_dim']

    def get_parameters(self):
        return self.ex_dict['dim_1'] * self.ex_dict['dim_2'] * self.ex_dict['num_weight']

class virtual_mlp_operation(virtual_basic_operation):
    def __init__(self, dim, ex_dict={}):
        super().__init__(dim=dim, ex_dict=ex_dict)

    def get_parameters(self):
        return self.ex_dict['dim_1'] * self.ex_dict['dim_2'] * self.ex_dict['num_weight']
