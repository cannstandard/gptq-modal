import sys
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from modelutils import find_layers
from quant import make_quant

# https://github.com/thisserand/FastChat/blob/main/fastchat/serve/load_gptq_model.py
def load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], kernel_switch_threshold=128):
    config = AutoConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize, faster=faster_kernel, kernel_switch_threshold=kernel_switch_threshold)

    del layers
    
    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model

# https://github.com/thisserand/FastChat/blob/main/fastchat/serve/load_gptq_model.py
def load_quantized(model_name, wbits=4, groupsize=128, threshold=128):
    model_name = model_name.replace('/', '_')
    path_to_model = Path(f'/FastChat/models/{model_name}')
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) == 1:
        pt_path = found_safetensors[0]

    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()

    model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize, kernel_switch_threshold=threshold)

    return model

# https://github.com/thisserand/FastChat/blob/main/fastchat/serve/cli.py
@torch.inference_mode()
def generate_stream(tokenizer, model, params, device,
                    context_len=2048, stream_interval=2):
    """Adapted from fastchat/serve/model_worker.py::generate_stream"""

    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values