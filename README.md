## The Intelligent Social Event Observer: Multi-Source Continuous Event Integration, Discovery, and Synopsis Generation with LLMs


## ✒️ Overview
SEO is a novel framework to integrate evolving events, detect emerging events, and automatically generate event synopsis for social networks in real time.

## 👇 Dataset
We evaluate our method for event detection on the [News14]() and [WCEP19]() datasets. The process files are in the path ./data/processed.

We evaluate our method for synopsis generation on the News14-detail, WCEP19-detail, News14-integrity, and WCEP19-integrity datasets. The datasets are in the path ./data/evaluation.


## ⏰ Quick Start

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


## 📝 Citation
Please cite our repository if you use SEO in your work.
```bibtex
```
