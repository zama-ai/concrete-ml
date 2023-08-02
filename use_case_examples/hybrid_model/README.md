# Hybrid model

This use case example showcases how to partially run layers in FHE.

In this case we apply a fully connected layer of a GPT-2 model in FHE.

## How to run this use-case

0. Install additional requirements using `python -m pip install -r requirements.txt`
1. Compile GPT-2 model using `python compile_hybrid_llm.py` script
1. Run FHE server using `bash serve.sh`
1. Run FHE client using `python infer_hybrid_llm_generate.py`
   - You will first be asked about the number of tokens that you want to generate
   - Then you will be able to enter your prompt
