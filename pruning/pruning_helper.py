import torch
from tqdm.auto import tqdm
import time
from torch.cuda.amp import GradScaler
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast
import os
import math
from transformers.models.llama.modeling_llama import LlamaRMSNorm

class collect_info_reg_llama(nn.Module):
    def __init__(self, model, p=None, lam=4.0):
        super(collect_info_reg_llama, self).__init__()
        self.sum_ori_params = 0 
        self.p = p  
        self.lam = lam  
        self.in_dim_list = [] 
        self.out_dim_list = []  
        self.num_w_list = []  
        self.structures = []  
        self.gate_type = []  
        
        modules = list(model.modules())  
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if type(m).__name__ == 'virtual_block_basic_operation':
                self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append('mlp_block')
            if type(m).__name__ == 'virtual_mlp_operation':
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                self.num_w_list.append(m.ex_dict['num_weight'])
                self.structures.append(m.dim)
                self.gate_type.append('mlp')
            if type(m).__name__ == 'virtual_block_attn_operation':
                ori_param = m.get_parameters()
                self.sum_ori_params += ori_param
                self.in_dim_list.append(m.ex_dict['dim_1'])
                self.out_dim_list.append(m.ex_dict['dim_2'])
                self.num_w_list.append(m.ex_dict['num_weight'])
                self.structures.append(m.dim)
                self.head_dim = m.head_dim
                self.num_heads = m.dim
                self.gate_type.append('attn_block')
            if type(m).__name__ == 'virtual_basic_operation':
                self.structures.append(m.dim)
                self.in_dim_list.append(None)
                self.out_dim_list.append(None)
                self.num_w_list.append(None)
                self.gate_type.append('basic_gate')

            print("Number of original parameters: %.3f" % (self.sum_ori_params / 10 ** 6))
            
    def forward(self, vectors):
        block_mlp_dim = None
        sum_params = 0
        i = 0
        while i < len(self.structures):
            # Process attention blocks
            if self.gate_type[i] == 'attn_block':
                attn_in_dim = vectors[i].sum()
                attn_out_dim = vectors[i+1].sum()
                current_params = attn_in_dim * 3 * self.out_dim_list[i] + attn_out_dim * self.out_dim_list[i]
                i += 2
                sum_params += current_params

            # Process MLP blocks
            if self.gate_type[i] == 'mlp_block':
                block_mlp_in_dim = vectors[i].sum()
                block_mlp_middle_dim = vectors[i+1].sum()
                block_mlp_out_dim = vectors[i+2].sum()
                current_params = block_mlp_in_dim * block_mlp_middle_dim * 2 + block_mlp_middle_dim * block_mlp_out_dim
                i += 3
                sum_params += current_params

        # Calculate parameter ratio
        param_ratio = sum_params / self.sum_ori_params
        if param_ratio > self.p:
            clamped_p_ratio = torch.clamp(param_ratio, min=self.p)
            loss = torch.log(clamped_p_ratio / self.p)
        else:
            clamped_p_ratio = torch.clamp(param_ratio, max=self.p)
            loss = torch.log(self.p / clamped_p_ratio)

        return self.lam * loss

class help_functions_hn(nn.Module):
    def __init__(self, structures, constrained=None):
        self.structures = structures
        self.constrained = constrained

    # Print the structures and summed values of gate vectors
    def print_info(self, vectors):
        print(self.structures)
        config = []
        for i in range(len(vectors)):
            config.append(vectors[i].sum().item())
        print(config)

    # Set gate vectors for different modules in the model
    def set_gate_vectors(self, model, vectors):
        if self.constrained == 'structural':
            modules = list(model.modules())
            ind = 0
            model_dim = vectors[0]
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == 'virtual_basic_operation':
                    m.set_vector_value(model_dim)
                if type(m).__name__ == 'virtual_att_operation':
                    m.set_vector_value(vectors[ind+1])
                    ind += 1
                if type(m).__name__ == 'virtual_mlp_operation':
                    m.set_vector_value(vectors[ind+1])
                    ind += 1
        elif self.constrained == 'same':
            modules = list(model.modules())
            ind = 0
            model_dim = vectors[0]
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == 'virtual_basic_operation':
                    m.set_vector_value(model_dim)
                if type(m).__name__ == 'virtual_block_basic_operation':
                    m.set_vector_value(model_dim)
                if type(m).__name__ == 'virtual_mlp_operation':
                    m.set_vector_value(vectors[ind+1])
                    ind += 1
                if type(m).__name__ == 'virtual_block_attn_operation':
                    m.set_vector_value(model_dim)
        else:
            modules = list(model.modules())
            ind = 0
            for layer_id in range(len(modules)):
                m = modules[layer_id]
                if type(m).__name__ == 'virtual_basic_operation':
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == 'virtual_block_basic_operation':
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == 'virtual_mlp_operation':
                    m.set_vector_value(vectors[ind])
                    ind += 1
                if type(m).__name__ == 'virtual_block_attn_operation':
                    m.set_vector_value(vectors[ind])
                    ind += 1

    def set_gate_status(self, model, use_gate=False):
        modules = list(model.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if hasattr(m, 'use_gate'):
                m.use_gate = use_gate
