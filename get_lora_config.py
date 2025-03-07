from peft import LoraConfig, get_peft_model
from peft import PeftModel
import torch

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
def get_lora_config(model, lora_rank, lora_alpha, lora_dropout):

    lora_config = LoraConfig(
        r=lora_rank,  
        lora_alpha=lora_alpha, 
        target_modules=find_all_linear_names(model),  
        lora_dropout=lora_dropout  
    )
    return lora_config
