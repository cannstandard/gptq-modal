from gptq import ModalGPTQ, stub
import pandas as pd

# For local testing, run `modal run -q run-interview.py --input questions.csv --temperature 0.7`
@stub.local_entrypoint()
def main(input: str, temperature: float):
    model = ModalGPTQ()
    questions = pd.read_csv(input)
    custom_precise = model.params(temperature=temperature, repetition_penalty=1.176, top_k=40, top_p=0.1)

    for idx, question in questions.iterrows():
            print("Q["+str(idx)+"]", question['name'])

            answer = ""
            for val in model.generate.call(question['prompt'], params=custom_precise):
                answer += val
                print(val, end="", flush=True)

            with open(f'answer_{idx}.txt', 'w') as f:
                f.write(answer)