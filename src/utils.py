import torch
import numpy as np
from external_libraries import b3
from sklearn import metrics
from scipy.stats import entropy
from transformers import pipeline
import yaml
import math
from args import Args
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from collections import Counter
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

args = Args.args

def get_llm_prompts(task_type, inputs):
    with open("/data/zhangruwen/Story2/config/prompts.yaml", 'r', encoding='utf-8') as file:
        prompts = yaml.load(file, yaml.FullLoader)

    if task_type == 'summary':  # 生成新闻组的摘要
        template = '''
            Articles: {articles}
            Output:
            '''
        content = prompts['summary']['user'] + template.format(articles=inputs)  # inputs type: list

    elif task_type == 'refresh_summary':  # 更新新闻组的摘要，更新到dict中
        template = '''
            Article: {article}
            Summary: {summary}
            Output:
            '''
        content = prompts['refresh_summary']['user'] + template.format(article=inputs["article"], summary=inputs["summary"])  # inputs type: dict

    elif task_type == 'add_summary':  # 生成某个新闻的摘要，添加到dict中
        template = '''
            Article: {article}
            Output:
            '''
        content = prompts['add_summary']['user'] + template.format(inputs)  # inputs type: dict

    elif task_type == 'abstract':  # 生成某个新闻的摘要
        template = '''
            Title: {title}
            Article: {article}
            Output:
            '''
        content = prompts['abstract']['user'] + template.format(title=inputs['title'], article=inputs['article'])  # inputs type: dict

    elif task_type == 'comparison':  # 比较是否是新的类
        template = '''
            Article List A: {listA}
            Article List B: {listB}
            Article: {article}
            Output:
            '''
        content = prompts['comparison']['user'] + template.format(listA=inputs['listA'], listB=inputs['listB'], article=inputs['article'])  # inputs type: dict

    elif task_type == 'judgement':  # 判断是否是新的类
        template = '''
            Article: {article}
            Summary: {summary}
            Output:
            '''
        content = prompts['judgement']['user'] + template.format(summary=inputs['summary'], article=inputs['article'])  # inputs type: dict

    return content

def get_llm_output(content, pipe, model_type=args.LLM_mode):
    with open("./config/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.load(file, yaml.FullLoader)

    if model_type == 'gpt-4o-mini':
        gpt_model = OpenAI(base_url=config["openai"]["base_url"], api_key=config["openai"]["api_key"])
        message = [{"role": "user", "content": content}]
        response = gpt_model.chat.completions.create(
            model=model_type,
            messages=message,
            temperature=config["openai"]["temperature"],
            max_tokens=config["openai"]["max_tokens"])
        return response.choices[0].message.content

    elif model_type == 'deepseek-qwen' or model_type == 'deepseek-llama' or model_type == 'llama3.2-3b' or model_type == 'llama3.1-8b':
        messages = [{"role": "user", "content": content}]
        outputs = pipe(messages, max_new_tokens=2048)
        return outputs[0]["generated_text"][-1]['content']

    elif model_type == 'gemini-pro':
        genai.configure(api_key=config["gemini"]["GOOGLE_API_KEY"], transport='rest')
        genai_model = genai.GenerativeModel(model_name=model_type)
        response = genai_model.generate_content(content)
        return response.text

    elif model_type == 'gemma-7b':
        pipe = pipeline("text-generation", model="google/gemma-7b", device="cuda")  # replace with your own model path
        message = [{"role": "user", "content": content}]
        response = pipe(message,
            max_new_tokens=config["gemma"]["max_tokens"],
            temperature=config["gemma"]["temperature"],
            top_k=config["gemma"]["top_k"],
            top_p=config["gemma"]["top_p"],
            do_sample=True,)
        return response[0]["generated_text"][-1]["content"]
    return None

# 判断是否是新的类
def judge_class(max_sim, max_sim_index, re_summary_dict, article, tuned_centers, slide_index, pipe):
    if max_sim < 0:  # 新类
        re_summary_dict[len(tuned_centers)] = article
        return -1, re_summary_dict

    summary_list = re_summary_dict[max_sim_index]  # 取出topk的类标签的summary
    if max_sim > args.thred:  # 旧类
        inputs_re = {"summary": summary_list, "article": article}
        prompt_re = get_llm_prompts('refresh_summary', inputs_re)
        output_re = get_llm_output(prompt_re, pipe)
        re_summary_dict[max_sim_index] = output_re
        return max_sim_index, re_summary_dict
    else:  # 旧类
        # (3)问询LLM判断==>获得新类/旧类
        inputs_judg = {"summary": summary_list, "article": article}
        prompt_judg = get_llm_prompts('judgement', inputs_judg)
        output_judg = get_llm_output(prompt_judg, pipe).lower()
        # (4)更新summary, 返回新类label和更新的summary
        if output_judg[0:3] == 'yes':  # 不是新类, 更新re_summary_dict
            inputs_re = {"summary": summary_list, "article": article}
            prompt_re = get_llm_prompts('refresh_summary', inputs_re)
            output_re = get_llm_output(prompt_re, pipe)
            re_summary_dict[max_sim_index] = output_re
            return max_sim_index, re_summary_dict
        else:  # 是新类, 增加一个新的项summary_dict
            re_summary_dict[len(tuned_centers)] = article
            return -1, re_summary_dict


# 使用KBERT提取关键词（获得当前时间窗口的关键字）
kw_model = KeyBERT(model='/data/zhangruwen/premodel/all-roberta-large-v1')
def extract_keywords_kbert(docs):
    kw_res = set()
    for doc in docs:
        # 为使结果多样化，使用最大边界相关算法(MMR)
        doc = " ".join(doc)
        keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english',
                                             use_mmr=True, diversity=0.7)
        for word in keywords:
            kw_res.add(word[0])

    return " ".join(kw_res)


# 使用TF-IDF提取关键词（获得当前时间窗口的关键字）
def extract_keywords_tfidf(documents, top_n=20):
    # 将过滤后的文档集转化为文本列表
    documents_text = [' '.join(document) for document in documents]

    # 创建TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer()

    # 计算TF-IDF权重
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents_text)

    # 获取特征词列表
    features = tfidf_vectorizer.get_feature_names()
    # features = tfidf_vectorizer.get_feature_names_out()

    # 计算每个文档中的关键词
    top_keywords_per_document = [] # 2d-list
    for doc_id in range(len(documents)):
        document_tfidf_weights = tfidf_matrix[doc_id].toarray()[0]
        top_keyword_indices = document_tfidf_weights.argsort()[-top_n:][::-1]
        top_keywords = [features[idx] for idx in top_keyword_indices]
        top_keywords_per_document.append(top_keywords)

    # 合并去重
    all_words = []
    for lst in top_keywords_per_document:
        all_words.extend(lst)
    word_counts = Counter(all_words)
    result = ' '.join(sorted(word_counts))

    return result

# 嵌入
st_model = SentenceTransformer('all-roberta-large-v1').cuda()
def embedding(texts):
    embedding = st_model.encode(texts)
    return embedding

def masking(org_embd, num_sens=1):
    mask = torch.ones(num_sens)
    mask[:org_embd.shape[0]] = 0

    return mask

# 损失函数
def infonce_loss(sample_outputs, class_indices, class_embds, temp=0.2):
    loss = 0
    for i in range(len(sample_outputs)):
        exp_temp_sims = torch.exp(torch.nn.functional.cosine_similarity(sample_outputs[i], class_embds) / temp)
        loss += -1 * torch.log(exp_temp_sims[class_indices[i]] / torch.sum(exp_temp_sims))
    return loss



# 评估指标
def eval_metric(label, cluster):
    # nmi = np.round(metrics.normalized_mutual_info_score(label, cluster),3)
    # ri = np.round(metrics.rand_score(label, cluster),3)
    ami = np.round(metrics.adjusted_mutual_info_score(label, cluster), 3)
    ari = np.round(metrics.adjusted_rand_score(label, cluster), 3)
    fscore, precision, recall = [np.round(k, 3) for k in b3.calc_b3(label, cluster)]

    return [precision, recall, fscore, ami, ari]


def adjust_learning_rate(optimizer, epoch, num_epochs, start_lr=0.001, min_lr=0.00001, warmup_epochs=20):
    if epoch < warmup_epochs:
        lr = start_lr * (epoch + 1) / warmup_epochs
    else:
        lr = min_lr + (start_lr - min_lr) * 0.5 * (
                1
                + math.cos(
            math.pi
            * (epoch - warmup_epochs)
            / (num_epochs - warmup_epochs)
        )
        )
    # print("Current Learning Rate: ", lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr




