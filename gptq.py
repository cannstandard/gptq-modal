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

stub = Stub(name="manticore-nochat")
MODEL_NAME = "TheBloke/Manticore-13B-GPTQ"

def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        local_dir=Path("/FastChat", "models", MODEL_NAME.replace('/', '_')),
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
        "git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda /FastChat/repositories/GPTQ-for-LLaMa",
        "cd /FastChat/repositories/GPTQ-for-LLaMa && pip install -r requirements.txt && python setup_cuda.py install",
        gpu="any",
    )
    .run_function(download_model)
)

if stub.is_inside(stub.vicuna_image):
    t0 = time.time()
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    import sys
    sys.path.insert(0, str(Path("/FastChat/repositories/GPTQ-for-LLaMa")))
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

        params = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": 1.1,
            "max_new_tokens": 512,
            "stop": self.sep,
        }

        print(prompt)

        prev = len(prompt) + 1
        count = 0
        for outputs in gptq_wrapper.generate_stream(self.tokenizer, self.model, params, "cuda"):
            yield outputs[prev:]
            prev = len(outputs)
            count = count + 2 # stream_interval
        dur = time.time() - t0

        print(f"{count} tokens generated in {dur:.2f}s, {1000*dur/count:.2f} ms/token", file=sys.stderr)

# For local testing, run `modal run -q gptq.py --input "Where is the best sushi in New York?"`
@stub.local_entrypoint()
def main(input: str):
    model = Vicuna()

    for val in model.generate.call(input):
        print(val, end="", flush=True)
