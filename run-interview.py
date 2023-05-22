from gptq import Vicuna
import pandas as pd

# For local testing, run `modal run -q src.llm_vicuna --input "Where is the best sushi in New York?"`
@stub.local_entrypoint()
def main(input: str):
    model = Vicuna()
    questions = pd.read_csv(input)

    for idx, question in questions.iterrows():
        for temp in [0.0, 0.7]:
            print("Q["+str(idx)+"]", question['name'], question['language'], 'temp=', temp)

            manticore_input = "### Instruction: "+question['prompt']+"\n\n### Assistant: "
            answer = ""
            for val in model.generate.call(manticore_input, temp):
                answer += val
                print(val, end="", flush=True)

            with open(f'answer_{idx}_{temp}.txt', 'w') as f:
                f.write(answer)