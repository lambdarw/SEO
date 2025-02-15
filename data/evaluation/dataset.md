# Event Evaluation Dataset

**_🐳The whole data will be released upon publication._**

## Data Collection
We used GPT-4o-mini to generate the single-choice QA datasets for News14 and WCEP19 to assess the quality of the synopsis.  
- For the article details evaluation, we generated one question for each news article.  
- For the event integrity evaluation, we generated five descriptive questions for each event.

Here is the Statistic of the Single Choice Evaluation Dataset. (# is the symbol of `the number of').
| Type | News14 | WCEP19 |
| :---: | :---: | :---: |
| # Synopsis Detail  | 16,136  | 29,931  |
| # Synopsis Integrity  | 3,940  | 2,595  |
| # Total  | 16,924  | 32,526  |

## Data Files
```md
data/
├── evaluation/
│   ├── details/           # QA datasets for evaluating article details
│   ├── outline/           # QA datasets for evaluating event integrity
│   └── dataset.md         # Dataset Introduction
└── processed/             # Data processing files
```

## Data Prompt
**Detail question generation:**
```md
detail_qa_generation:
  user: |-
    Based on the given article, generate a single question along with its correct answer and one plausible distractor. 
    1. The question must be clear and fully answerable using only the article. Sub-questions are strictly prohibited.
    2. The answer should be concise and comprehensible.
    3. Avoid the following types of questions: questions requiring numerical reasoning; questions requiring significant external world knowledge; and questions requiring inference beyond the text.~Please follow the above rules. 
    You MUST only respond in the dictionary format: {"Question": "xxx?", "Answer": "xxx", "Noising_answers": "xxx"}"
    DO NOT INCLUDE ANYTHING ELSE.
```

**Outline question generation:**
```md
outline_qa_generation:
  user: |-
    Based on the given summary, generate five distinct questions, each accompanied by one correct answer and one plausible distractor. 
    1. The question must be clear and fully answerable using only the summary. Sub-questions are strictly prohibited.
    2. The answer should be concise and comprehensible.
    3. Avoid the following question types: questions requiring numerical reasoning; questions requiring significant external world knowledge; and questions requiring inference beyond the text.~Please follow the above rules. 
    You MUST only respond in the list format: [{"Question": "xxx?", "Answer": "xxx", "Noising Answers": "xxx"}, {...}, ...] 
    DO NOT INCLUDE ANYTHING ELSE.
```
