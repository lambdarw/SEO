## The Intelligent Social Event Observer: Multi-Source Continuous Event Integration, Discovery, and Synopsis Generation with LLMs


## ✒️ Overview
SEO is a novel framework to integrate evolving events, detect emerging events, and automatically generate event synopsis for social networks in real time.

## 👇 Dataset
We evaluate our method for event detection on the [News14]() and [WCEP19]() datasets. The process files are in the path ./data/processed.

We evaluate our method for synopsis generation on the News14-detail, WCEP19-detail, News14-integrity, and WCEP19-integrity datasets. The datasets are in the path ./data/evaluation.


## ✨ Quick Start

**Step1: Write a configuration file in YAML format**

Users can easily configure the parameters of LLMs in a YAML file. 
The path of the configuration file is SEO/config/config.yaml

```yaml
openai:
  api_key: # your_openai_API_key
  base_url: 
  temperature: 0.2  
  max_tokens: 2048
llama:
  temperature: 0.2
  max_tokens: 2048
prompts:
  prompt: config/prompts.yaml
```

**Step2: Running the model**
```python
python main.py
```

## ⏰ Optimized Inference with vLLM
To maximize inference efficiency, we recommend leveraging the [vLLM](https://docs.vllm.ai/en/latest/) framework for batched and cached inference. 

Installation Instructions, requiring CUDA 11.8+:
```yaml
pip install torch==2.3.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install xformers==0.0.26.post1 vllm==0.5.1+cu118
```

## 📖 Citation
Please cite our repository if you use SEO in your work.
```bibtex
```
