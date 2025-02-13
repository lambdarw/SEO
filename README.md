## The Intelligent Social Event Observer: Multi-Source Continuous Event Integration, Discovery, and Synopsis Generation with LLMs


## Overview
SEO is a novel framework to integrate evolving events, detect emerging events, and automatically generate event synopsis for social networks in real time. It supports refining the event detection range by adjusting the size of the time window.


### Dataset
We evaluate our method on the [News14]() and [WCEP19]() dataset.


## Quick Start

**Step1: Write a configuration file in yaml format**

Users can easily configure the parameters of LLMs in a yaml file. 
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


## Citation
Please cite our repository if you use SEO in your work.
```bibtex
```
