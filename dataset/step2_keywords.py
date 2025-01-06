import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from tqdm import tqdm
from keybert import KeyBERT
import re

def remove_digits_and_spaces(s):
    # 移除数字
    s = re.sub(r'\d', '', s)
    # 移除连续空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def masking(df, idx, num_sens=1):
    org_embd = torch.tensor([df.loc[idx, 'summary_embds']])
    maksed_embd = torch.zeros(num_sens, org_embd.shape[1])
    mask = torch.ones(num_sens)
    maksed_embd[:org_embd.shape[0], :] = org_embd
    mask[:org_embd.shape[0]] = 0

    return maksed_embd, mask

# 使用KBERT提取关键词
kw_model = KeyBERT(model='all-roberta-large-v1')
st_model = SentenceTransformer('all-roberta-large-v1').cuda()

def extract_keywords_kbert(doc):
    kw_res = set()
    # 为了使结果多样化，使用最大边界相关算法(MMR)
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                                        use_mmr=True, diversity=0.7, top_n=10)
    # for word in keywords:
    #    kw_res.add(word[0])

    # 过滤掉形容词、动词和停用词
    for word in keywords:
        word = remove_digits_and_spaces(word[0])  # 去掉数字和连续空格
        if len(word) != 0:
            kw_res.add(word)
            # w = nlp(word)
            # if w.pos_ not in {'ADJ', 'VERB'}:
            #     kw_res.add(word[0])

    return list(kw_res)[:5]


if __name__ == "__main__":
    DATASET_NAME = 'WCEP18'  # News14  WCEP18  WCEP19
    nlp = spacy.load("en_core_web_lg")
    # 读json
    article_df = pd.read_json(DATASET_NAME+"/"+DATASET_NAME + "_step1_summary.json")

    # 提取关键字
    keywords = []
    # keyword_embeddings = []
    summary_embeddings = []
    for summary in tqdm(article_df['summary']):
        keyword = extract_keywords_kbert(summary)
        # keyword_embedding = st_model.encode(keyword)
        summary_embedding = st_model.encode(summary)

        keywords.append(keyword)
        # keyword_embeddings.append(keyword_embedding)
        summary_embeddings.append(summary_embedding)

    # keyword_embeddings = np.array(keyword_embeddings)
    # keyword_embeddings = torch.tensor(keyword_embeddings)

    # summary的嵌入向量
    article_df['summary_embds'] = summary_embeddings
    masked = [masking(article_df, idx) for idx in article_df.index]
    masked_tensors = torch.stack([m[0] for m in masked])
    masks = torch.stack([m[1] for m in masked])

    # 生成新文件进行保存
    article_df['keywords'] = keywords
    article_df[['id', 'date', 'title', 'sentence_counts', 'summary', 'query', 'keywords']].to_json(
        DATASET_NAME+"/"+DATASET_NAME + "_step2_summary_kw.json")  # remove 'story' or 'query' if not available
    # torch.save(keyword_embeddings, DATASET_NAME+"/"+DATASET_NAME + "_kw_embds.pt")
    torch.save(masked_tensors, DATASET_NAME + "/" + DATASET_NAME + "_masked_embds_sample_summary.pt")
    torch.save(masks, DATASET_NAME + "/" + DATASET_NAME + "_masks_sample_summary.pt")
