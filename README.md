# GPTQ-on-Modal

Based on https://github.com/modal-labs/quillman

Try it with `modal run -q gptq.py --input "What kind of alien was Worf?"`

## What's new?

* Upgrade base image to NVIDIA Toolkit 11.7.1
* FastChat dependancy removed (along with all its ugly path hacks)
* Added repeat_penalty, topk, topp