from peft import LoraConfig, get_peft_model, PeftModel
import torch


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

def create_lora(model, args):
    lora_config = LoraConfig(
        r=args.lora_rank,  # Rank of LoRA
        lora_alpha=args.lora_alpha,  # Scaling factor
        target_modules=find_all_linear_names(model),  # Target modules, can vary depending on architecture
        lora_dropout=args.lora_dropout  # Dropout for LoRA layers
    )
    model = get_peft_model(model, lora_config)
    print("LoRA modules added to the model.")


    return model