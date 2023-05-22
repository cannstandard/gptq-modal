"""
Vicuna 13B language model, 4-bit quantized for faster inference.
Adapted from https://github.com/thisserand/FastChat.git

Path to weights provided for illustration purposes only,
please check the license before using for commercial purposes!
"""
import time
from pathlib import Path
from modal import Image, Stub, method, create_package_mounts
import pandas as pd

stub = Stub(name="manticore-newcuda")
MODEL_NAME = "TheBloke/Manticore-13B-GPTQ"

def download_model():
    from huggingface_hub import snapshot_download

    # Match what FastChat expects
    # https://github.com/thisserand/FastChat/blob/4a57c928a906705404eae06f7a44b4da45828487/download-model.py#L203
    output_folder = f"{'_'.join(MODEL_NAME.split('/')[-2:])}"

    snapshot_download(
        local_dir=Path("/FastChat", "models", output_folder),
        repo_id=MODEL_NAME,
        allow_patterns="*"
    )


stub.vicuna_image = (
    Image.from_dockerhub(
        "nvidia/cuda:11.7.1-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update",
            "RUN apt-get install -y python3 python3-pip python-is-python3 git build-essential",
        ],
    )
    .run_commands(
        "git clone https://github.com/thisserand/FastChat.git",
        "cd FastChat && pip install -e .",
    )
    .run_commands(
        # FastChat hard-codes a path for GPTQ, so this needs to be cloned inside repositories.
        "git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda /FastChat/repositories/GPTQ-for-LLaMa",
        "cd /FastChat/repositories/GPTQ-for-LLaMa && python setup_cuda.py install",
        gpu="any",
    )
    .run_function(download_model)
)

if stub.is_inside(stub.vicuna_image):
    t0 = time.time()
    import os
    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )

    # This version of FastChat hard-codes a relative path for the model ("./model"),
    # making this necessary :(
    os.chdir("/FastChat")
    import gptq_wrapper

@stub.cls(image=stub.vicuna_image, gpu="A10G", container_idle_timeout=300, mounts=create_package_mounts(["gptq_wrapper"]))
class Vicuna:
    def __enter__(self):
        tokenizer = gptq_wrapper.AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Loading GPTQ quantized model...")
        model = gptq_wrapper.load_quantized(MODEL_NAME)
        model.cuda()

        self.model = model
        self.tokenizer = tokenizer
        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.configure()

    def configure(self,
                  system="A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the human's questions.",
                  human='Instruction',
                  ai='Assistant',
                  sep='###'):
        self.system = system
        self.human = human
        self.ai = ai
        self.sep = sep

    def prompt(self, messages):
        ret = self.system + self.sep
        for role, message in messages:
            if message:
                ret += role + ": " + message + self.sep
            else:
                ret += role + ":"
        return ret

    @method()
    async def generate(self, input, temperature = 0.0, history=[]):
        if input == "":
            return

        t0 = time.time()
        assert len(history) % 2 == 0, "History must be an even number of messages"

        messages = []
        for i in range(0, len(history), 2):
            messages.append((self.human, history[i]))
            messages.append((self.ai, history[i + 1]))
        messages.append((self.human, input))
        messages.append((self.ai, None))

        prompt = self.prompt(messages)
        print("P:", prompt)

        params = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": 512,
            "stop": self.sep,
        }

        prev = len(prompt) + 2
        count = 0
        for outputs in gptq_wrapper.generate_stream(self.tokenizer, self.model, params, "cuda"):
            yield outputs[prev:].replace("##", "")
            prev = len(outputs)
            count = count + 2 # stream_interval
        dur = time.time() - t0

        print("")
        print(f"{count} tokens generated in {dur:.2f}s, {1000*dur/count:.2f} ms/token")


# For local testing, run `modal run -q src.llm_vicuna --input "Where is the best sushi in New York?"`
@stub.local_entrypoint()
def main(input: str):
    model = Vicuna()

    for val in model.generate.call(input):
        print(val, end="", flush=True)
