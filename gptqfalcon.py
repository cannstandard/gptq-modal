"""
GPTQ wrapper for 13B language model, 4-bit quantized for faster inference.
Adapted from https://github.com/thisserand/FastChat.git and https://github.com/lm-sys/FastChat.git

Path to weights provided for illustration purposes only, please check the license before using for commercial purposes!
"""
import time
from pathlib import Path
from modal import Image, Stub, method, create_package_mounts, gpu

#MODEL_NAME = "TheBloke/falcon-40b-instruct-GPTQ"
MODEL_NAME = "TheBloke/falcon-7b-instruct-GPTQ"
MODEL_FILES = ["*"]

stub = Stub(name=MODEL_NAME.replace('/', '-'))

#### NOTE: Modal will not rebuild the container unless this function name or it's code contents change.
####       It is NOT sufficient to change any of the constants above.
def download_falcon_7b_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        local_dir=Path("/model"),
        repo_id=MODEL_NAME,
        allow_patterns=MODEL_FILES
    )

stub.gptq_image = (
    Image.from_dockerhub(
        "nvidia/cuda:11.7.1-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3 git build-essential",
        ],
    )
    .run_commands(
        "git clone https://github.com/PanQiWei/AutoGPTQ /repositories/AutoGPTQ",
        "cd /repositories/AutoGPTQ && pip install . && pip install einops && python setup.py install",
        gpu="any",
    )
    .run_function(download_falcon_7b_model)
)

if stub.is_inside(stub.gptq_image):
    t0 = time.time()
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    import sys
    sys.path.insert(0, str(Path("/repositories/AutoGPTQ")))
    import torch
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM



@stub.cls(image=stub.gptq_image, gpu=gpu.A10G(count=1), concurrency_limit=1, container_idle_timeout=300)
class ModalFalconGPTQ:
    def __enter__(self):
        quantized_model_dir = "/model"
        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=False)
        print('Loading model...')
        model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, use_safetensors=True, torch_dtype=torch.float32, trust_remote_code=True)
        self.model = model
        self.tokenizer = tokenizer
        print(f"Model loaded in {time.time() - t0:.2f}s")

    @method()
    def generate(self, prompt):
        prompt_template = f"### Instruction: {prompt}\n### Response:"

        tokens = self.tokenizer(prompt_template, return_tensors="pt").to("cuda:0").input_ids
        output = self.model.generate(input_ids=tokens, max_new_tokens=512, do_sample=True, eos_token_id=11, temperature=0.8)
        return self.tokenizer.decode(output[0])

# For local testing, run `modal run -q gptq.py --input "Where is the best sushi in New York?"`
@stub.local_entrypoint()
def main(input: str):
    model = ModalFalconGPTQ()
    print(model.generate.call(input))
