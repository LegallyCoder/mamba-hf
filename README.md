# mamba-hf
<img src="https://th.bing.com/th/id/OIG.Jp5dA01tOAFcwSp544nv?pid=ImgGn" width="300" height="300" alt="mamba-hf">

Implementation of the Mamba SSM with hf_integration.

# Usage:
To use the **mamba-hf**, follow these steps:

1. Clone the repository to your local machine.
   
```bash
git clone https://github.com/LegallyCoder/mamba-hf
```
2. Open a terminal or command prompt and navigate to the script's directory.
```bash
cd src
```

3. Install the required packages using this command:

```bash
pip3 install -r requirements.txt
```

4. Open new python file at the script's directory.
```python
from modeling_mamba import MambaForCausalLM
from transformers import AutoTokenizer

model = MambaForCausalLM.from_pretrained('Q-bert/Mamba-130M')
tokenizer = AutoTokenizer.from_pretrained('Q-bert/Mamba-130M')

text = "Hi"

input_ids = tokenizer.encode(text, return_tensors="pt")

output = model.generate(input_ids, max_length=20, num_beams=5, no_repeat_ngram_size=2)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

```
> Hi, I'm looking for a new job. I've been working at a company for about a year now.

# For more:
You can look at here 
[Mamba Models Collection](https://huggingface.co/collections/Q-bert/mamba-65869481595e25821853d20d)
## References and Credits:

The Mamba architecture was introduced in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor).

Thank for the simple implementation (https://github.com/johnma2006/mamba-minimal)

The official implementation is here: https://github.com/state-spaces/mamba/tree/main
