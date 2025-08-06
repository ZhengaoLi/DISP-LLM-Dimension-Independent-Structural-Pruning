from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import sys
import torch

from pruning.pruning_helper import collect_info_reg_llama, help_functions_hn
from pruning.hypernetwork import hypernetwork

from models.modeling_llama_pruning import LlamaForCausalLM


import time
import tqdm
import math

def evaluate(model, tokenizer, datasets="wiki", block_size=4096):
    model.eval()
    model.cuda()
    datasets = datasets

    total_toks = 0
   
    for dsname in datasets.split(","):
        test_string = load_eval_data(dsname)
        encoded_text = tokenizer.encode(
        test_string, return_tensors='pt')
        encoded_text = encoded_text[:, : 256 *  2048]


        nlls = 0
        toks = 0
        with torch.inference_mode():
            block_size = block_size 
            for i in tqdm.tqdm(range(0, encoded_text.shape[1], block_size)):
                inp = encoded_text[:, i : i + block_size]


                model_output = model(inp.cuda())
                if hasattr(model_output, 'logits'):
                    logits = model_output.logits
                else:
                    logits = model_output
                nll = torch.nn.functional.cross_entropy(
                logits[0,:-1], inp[0, 1:].to(dtype=torch.long).cuda(), reduction="sum")
                toks += inp.size(1) - 1
                nlls += nll.item()

        print(encoded_text.shape, logits.shape)
        encoded_text = encoded_text[:, : logits.shape[0]]
        ppl = math.exp(nlls / toks)
        print(f"Perplexity on {dsname}: {ppl:.2f}")
        total_toks += toks

def load_eval_data(dataset_name: str) -> str:
    # this mimics gptq datautils
    if dataset_name == "wikitext":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "ptb":
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testdata = "\n\n".join(testdata["sentence"])
    elif dataset_name == 'pile':
        testdata = load_hf_dataset_pile_dedup('validation')
        testdata = "\n\n".join(testdata["text"])
    elif dataset_name == "c4":
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testdata = " ".join(testdata[:1100]["text"])

    else:
        raise ValueError("invalid dataset name (wikitext, ptb, c4 are allowed)")
    return testdata

def main(
    hf_model: str = "meta-llama/Llama-2-7b-hf",
    hn_path: str = "to/your/hn_path/hn-ckpt-final-0.50.pt",
    out_dir: str = 'to/your/out_dir',
    evaluate_ppl: bool = False, 
    block_size=4096,
) -> None:

    model = LlamaForCausalLM.from_pretrained(hf_model)
    model.config.use_cache = False
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model.cuda()

    reg = collect_info_reg_llama(model, p=0.5)

    hn_helper = help_functions_hn(reg.structures, constrained=False)

    hn_stat_dict = torch.load(hn_path,  map_location=torch.device('cpu'))

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in hn_stat_dict.items():
        name = k.replace('module.', '')
        # k[7:] # remove `module.`
        new_state_dict[name] = v

    hn_stat_dict = new_state_dict

    hn = hypernetwork(t_structures = reg.structures)
    hn.cuda()
    hn.load_state_dict(hn_stat_dict)

    print(model)
    print(reg.structures)

    hn.eval()
    with torch.no_grad():
        vectors = hn()

    hn_helper.set_gate_vectors(model,vectors)

    i = 0
    modules = list(model.modules())
    for layer_id in range(len(modules)):
        m = modules[layer_id]
        
        if type(m).__name__ == 'LlamaMLP':
            in_vector = vectors[i]
            mid_vector = vectors[i+1]
            out_vector = vectors[i+2]

            in_dim = int(in_vector.sum().item())
            mid_dim = int(mid_vector.sum().item())
            out_dim = int(out_vector.sum().item())

            select_index = (in_vector==1).nonzero().squeeze()
            copy_index = (out_vector==1).nonzero().squeeze()
            mid_index = (mid_vector==1).nonzero().squeeze()

            pruned_gate_proj = torch.nn.Linear(in_features=in_dim, out_features=mid_dim, bias=False)
            pruned_up_proj = torch.nn.Linear(in_features=in_dim, out_features=mid_dim, bias=False)
            pruned_down_proj = torch.nn.Linear(in_features=mid_dim, out_features=out_dim, bias=False)

            pruned_gate_proj_weight = m.gate_proj.weight.data[mid_index,:]
            pruned_gate_proj_weight = pruned_gate_proj_weight[:,select_index]

            pruned_up_proj_weight = m.up_proj.weight.data[mid_index,:]
            pruned_up_proj_weight = pruned_up_proj_weight[:,select_index]

            pruned_down_proj_weight = m.down_proj.weight.data[copy_index,:]
            pruned_down_proj_weight = pruned_down_proj_weight[:,mid_index]

            pruned_gate_proj.weight.data.copy_(pruned_gate_proj_weight)
            pruned_up_proj.weight.data.copy_(pruned_up_proj_weight)
            pruned_down_proj.weight.data.copy_(pruned_down_proj_weight)

            if pruned_gate_proj.bias is not None:
                pruned_gate_proj.bias.data.copy_(m.gate_proj.bias.data[mid_index])
            if pruned_up_proj.bias is not None:
                pruned_up_proj.bias.data.copy_(m.up_proj.bias.data[mid_index])
            if pruned_down_proj.bias is not None:
                pruned_down_proj.bias.data.copy_(m.down_proj.bias.data[copy_index])


            m.gate_proj = pruned_gate_proj
            m.up_proj = pruned_up_proj
            m.down_proj = pruned_down_proj

            i = i+3
        
        if type(m).__name__ == 'LlamaAttention' or type(m).__name__ == 'LlamaFlashAttention2' or type(m).__name__ == 'LlamaSdpaAttention' :
            in_vector = vectors[i]
            out_vector = vectors[i+1]

            in_dim = int(in_vector.sum().item())
            out_dim = int(out_vector.sum().item())

            
            select_index = (in_vector==1).nonzero().squeeze()
            copy_index = (out_vector==1).nonzero().squeeze()

            # if type(m).__name__ == "Qwen2Attention" or type(m).__name__ == "Qwen2FlashAttention2": 
            #     pruned_q_proj = torch.nn.Linear(in_dim, m.num_heads*m.head_dim, bias=True)
            #     pruned_k_proj = torch.nn.Linear(in_dim, m.num_key_value_heads*m.head_dim, bias=True)
            #     pruned_v_proj = torch.nn.Linear(in_dim, m.num_key_value_heads*m.head_dim, bias=True)
            # else:
            pruned_q_proj = torch.nn.Linear(in_dim, m.num_heads*m.head_dim, bias=False)
            pruned_k_proj = torch.nn.Linear(in_dim, m.num_key_value_heads*m.head_dim, bias=False)
            pruned_v_proj = torch.nn.Linear(in_dim, m.num_key_value_heads*m.head_dim, bias=False)
            pruned_dense = torch.nn.Linear(m.num_heads*m.head_dim, out_dim, bias=False)

            q_weight = m.q_proj.weight.data[:, select_index]
            k_weight = m.k_proj.weight.data[:, select_index]
            v_weight = m.v_proj.weight.data[:, select_index]

            dense_weight = m.o_proj.weight.data[copy_index, :]

            pruned_q_proj.weight.data.copy_(q_weight)
            pruned_k_proj.weight.data.copy_(k_weight)
            pruned_v_proj.weight.data.copy_(v_weight)
            pruned_dense.weight.data.copy_(dense_weight)

            if pruned_q_proj.bias is not None:
                pruned_q_proj.bias.data.copy_(m.q_proj.bias.data)
            if pruned_k_proj.bias is not None:
                pruned_k_proj.bias.data.copy_(m.k_proj.bias.data)
            if pruned_v_proj.bias is not None:  
                pruned_v_proj.bias.data.copy_(m.v_proj.bias.data)
            if pruned_dense.bias is not None:
                pruned_dense.bias.data.copy_(m.o_proj.bias.data[copy_index])

            m.q_proj = pruned_q_proj
            m.k_proj = pruned_k_proj
            m.v_proj = pruned_v_proj
            m.o_proj = pruned_dense

            i = i+2

    model.cpu()
    state_dict = model.state_dict()
    
    from models.modeling_llama_pruned import LlamaForCausalLM as PrunedLlamaForCausalLM

    PrunedLlamaForCausalLM.cfgs = vectors 

    vector_list = []
    for i in range(len(vectors)):
        #vectors[i] = vectors[i].tolist()
        vector_list.append(vectors[i].tolist())

    config = model.config

    pruned_model = PrunedLlamaForCausalLM(config)
    pruned_model.load_state_dict(state_dict)
    print(pruned_model)

    sum_params = sum(p.numel() for p in pruned_model.parameters())
    print("number of pruned parameters: %.3f" % (sum_params / 10 ** 6))
    pruned_model.register_for_auto_class("AutoModelForCausalLM")

    pruned_model.save_pretrained(out_dir)
    hf_tokenizer.save_pretrained(out_dir)

    print('read file')
    with open(os.path.join(out_dir,'modeling_llama_pruned.py'), 'r') as f:
        lines = f.read().split('\n')
        # val = int(lines[14].split('=')[-1])
        new_line = '        self.cfgs = {}'.format(vector_list)
        new_file = '\n'.join(lines[:1162] + [new_line] +  lines[1163:])

    print('write to file')
    with open(os.path.join(out_dir,'modeling_llama_pruned.py'), 'w') as f:
        f.write(new_file)

    if evaluate_ppl:
        pruned_model.cuda()

        sum_params = sum(p.numel() for p in model.parameters())
        print("number of total parameters: %.3f" % (sum_params / 10 ** 6))

        torch.set_float32_matmul_precision("high")
        evaluate(pruned_model, hf_tokenizer, datasets='wikitext,ptb',block_size=block_size)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)