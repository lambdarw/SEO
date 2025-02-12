import pandas as pd
import spacy
from tqdm import tqdm
from utils import get_llm_prompts, get_llm_output
import json
import re


# Generate story synopsis
def generate_summary(title, article, task_type='summary'):
    inputs = {"title": title, "article": article}
    content = get_llm_prompts(task_type, inputs)
    g_summary = get_llm_output(content)
    return g_summary

# Data sampling
def df_sample(df, random_percentage=0.02):
    n_rows = len(df)
    n_sample = int(n_rows * random_percentage)
    n_sample = 2

    df_sample = df.head(n_sample)
    return df_sample


if __name__ == "__main__":
    sample_flag = False
    summary_flag = True
    DATASET_NAME = 'News14'  # News14  WCEP19
    article_df = pd.read_json(DATASET_NAME+"/"+DATASET_NAME + "_raw.json")

    if sample_flag:
        article_df = df_sample(article_df)

    '''1.Data processing'''
    nlp = spacy.load("en_core_web_lg")
    article_df.dropna(subset=['text', 'title'], inplace=True)
    # set corresponding column names. Drop 'story' or 'query' (used to collect stories) column if not available
    article_df.columns = ['id', 'date', 'title', 'text', 'query']
    article_df['sentences'] = [[t] for t in article_df.title]
    article_df['sentence_counts'] = ""

    '''2.Story Synopsis Extraction'''
    all_summary = []
    all_sentences_len = []
    j = 0
    print("\nData Processing...")
    for text in tqdm(article_df['text'].values):
        parsed = nlp(text)
        sentences = []
        for s in parsed.sents:
            if len(s) > 1:
                sentences.append(s.text)

        text = " ".join(sentences).replace("\n ", "")
        title = article_df.title[j]
        summary = generate_summary(title, text)

        all_summary.append(f"{title}: {summary}")
        all_sentences_len.append(len(sentences))

        j += 1
        # Saving files
        with open(DATASET_NAME + "/" + DATASET_NAME + "_all_summary.json", "w") as f:
            json.dump(all_summary, f)
        with open(DATASET_NAME + "/" + DATASET_NAME + "_all_sentences_len.json", "w") as f:
            json.dump(all_sentences_len, f)

    # Writing files
    article_df['summary'] = all_summary
    article_df['sentence_counts'] = all_sentences_len

    article_df['date'] = [str(k)[:10] for k in article_df['date']]  # Format and sort the date column
    article_df.sort_values(by=['date'], inplace=True)
    article_df.reset_index(inplace=True, drop=True)  # Reset the index and add the id column
    article_df['id'] = article_df.index
    article_df['story'] = [int(story_id) for story_id in article_df["story"]]

    article_df[['id', 'date', 'title', 'sentence_counts', 'summary', 'query']].to_json(
        DATASET_NAME + "/" + DATASET_NAME + "_step1_summary.json")  # remove 'story' or 'query' if not available
