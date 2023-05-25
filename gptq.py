"""
GPTQ wrapper for 13B language model, 4-bit quantized for faster inference.
Adapted from https://github.com/thisserand/FastChat.git and https://github.com/lm-sys/FastChat.git

Path to weights provided for illustration purposes only, please check the license before using for commercial purposes!
"""
import time
from pathlib import Path
from modal import Image, Stub, method, create_package_mounts, gpu

MODEL_NAME = "TheBloke/wizard-mega-13B-GPTQ"
MODEL_FILES = ["*.safetensors", "*.json", "*.model"]
MODEL_WBITS = 4
MODEL_GROUPSIZE = 128 # -1 to disable

stub = Stub(name="modalgptq-"+MODEL_NAME.replace('/', '-'))

def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        local_dir=Path("/FastChat", "models", MODEL_NAME.replace('/', '_')),
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
        "git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda /FastChat/repositories/GPTQ-for-LLaMa",
        "cd /FastChat/repositories/GPTQ-for-LLaMa && pip install -r requirements.txt && python setup_cuda.py install",
        gpu="any",
    )
    .run_function(download_model)
)

if stub.is_inside(stub.gptq_image):
    t0 = time.time()
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    import sys
    sys.path.insert(0, str(Path("/FastChat/repositories/GPTQ-for-LLaMa")))
    import gptq_wrapper

@stub.cls(image=stub.gptq_image, gpu=gpu.A10G(count=1), concurrency_limit=1, container_idle_timeout=300, mounts=create_package_mounts(["gptq_wrapper"]))
class ModalGPTQ:
    def __enter__(self):
        tokenizer = gptq_wrapper.AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Loading GPTQ quantized model...")
        model = gptq_wrapper.load_quantized(MODEL_NAME, wbits=MODEL_WBITS, groupsize=MODEL_GROUPSIZE)
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

    def params(self, temperature=0.7, repetition_penalty=1.0, top_k=-1, top_p=1.0):
        return {
            "model": MODEL_NAME,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "top_p": top_p
        }

    def prompt(self, messages):
        ret = self.system
        for role, message in messages:
            if message:
                ret += self.sep + ' ' + role + ": " + message 
            else:
                ret += self.sep + ' ' + role + ":"
        return ret

    @method()
    async def generate(self, input, max_new_tokens = 512, history=[], params=None):
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
        print(prompt)

        if params is None:
            params = self.params()
        params['prompt'] = prompt
        params['max_new_tokens'] = max_new_tokens
        params['stop'] = self.sep
        print(params)

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
    model = ModalGPTQ()

    # creative and precise settings from https://old.reddit.com/r/LocalLLaMA/wiki/index
    default = model.params()
    precise = model.params(temperature=0.7, repetition_penalty=1.176, top_k=40, top_p=0.1)
    creative = model.params(temperature=0.72, repetition_penalty=1.1, top_k=0, top_p=0.73)

    for val in model.generate.call(input, params=precise):
        print(val, end="", flush=True)
