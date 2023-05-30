# GPTQ-on-Modal

Based on https://github.com/modal-labs/quillman

* Upgrade base image to NVIDIA Toolkit 11.7.1
* FastChat dependancy removed (along with all its ugly path hacks)
* Added repeat_penalty, topk, topp

# Sample models

`gptq.py` runs [TheBloke/wizard-mega-13B-GPTQ](https://huggingface.co/TheBloke/wizard-mega-13B-GPTQ)

`gptq30b.py` runs [TheBloke/VicUnlocked-30B-LoRA-GPTQ](https://huggingface.co/TheBloke/VicUnlocked-30B-LoRA-GPTQ)

# Try it!

`modal run -q gptq.py --input "What kind of alien was Worf?"`
