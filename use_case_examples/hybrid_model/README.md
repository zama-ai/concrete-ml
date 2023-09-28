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

## Adapt to any LLM

This use case can easily be used with a different model than GPT2 with any other model.

For example, you can use [Phi 1.5](https://huggingface.co/microsoft/phi-1_5) on Hugging Face and protect the weights that are used to compute Query, Keys, Values as follows:

<!--pytest-codeblocks:skip-->

```bash
python compile_hybrid_llm.py --model-name microsoft/phi-1_5 --module-names layers.1.mixer.Wqkv
```
