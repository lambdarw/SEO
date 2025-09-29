import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
from tqdm import trange
import json
from collections import Counter
from transformers import pipeline
from args import Args
from utils import infonce_loss, eval_metric, embedding, judge_class, \
    get_llm_prompts, get_llm_output, masking, adjust_learning_rate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3, 5"

# 设置 transformers 的日志级别为 ERROR 或更高
logging.getLogger("transformers").setLevel(logging.ERROR)

class NewsAttentionModel(torch.nn.Module):
    def __init__(self, D_in, D_hidden, head, dropout=0.0):
        super(NewsAttentionModel, self).__init__()
        self.mha_news = torch.nn.MultiheadAttention(embed_dim=D_in, num_heads=head, dropout=dropout, batch_first=True)
        self.layernorm = torch.nn.LayerNorm(D_in)
        self.embd = torch.nn.Linear(D_in, D_hidden)
        self.attention = torch.nn.Linear(D_hidden, 1)
    def forward(self, news_content, mask=None):
        news_embd, _ = self.mha_news(news_content, news_content, news_content, key_padding_mask=mask)
        news_out = self.layernorm(news_content + news_embd)
        news_out = self.embd(news_out)
        news_out = torch.tanh(news_out)
        a = self.attention(news_out)
        if mask is not None:
            a = a.masked_fill_((mask == 1).unsqueeze(-1), float('-inf'))
        w = torch.softmax(a, dim=1)
        o1 = torch.matmul(w.permute(0, 2, 1), news_out)
        return o1

class KeywordAttentionModel(torch.nn.Module):
    def __init__(self, D_in, D_hidden, head, dropout=0.2):
        super(KeywordAttentionModel, self).__init__()
        self.mha_keywords = torch.nn.MultiheadAttention(embed_dim=D_in, num_heads=head, dropout=dropout,
                                                        batch_first=True)
        self.embd = torch.nn.Linear(D_in, D_hidden)
        self.attention = torch.nn.Linear(D_hidden, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()  # 可以根据需要调整

    def forward(self, keywords, news_content, mask=None):
        # 使用多头注意力机制
        kw_embd, _ = self.mha_keywords(keywords, news_content, news_content, key_padding_mask=mask)
        # 通过线性层和激活函数
        kw_out = self.dropout(self.activation(self.embd(kw_embd)))
        # 计算注意力权重
        a = self.attention(kw_out)
        w = torch.softmax(a, dim=1)
        # 计算加权输出
        o2 = torch.matmul(w.permute(0, 2, 1), kw_out)
        return o2

class Model(torch.nn.Module):
    def __init__(self, D_in, D_hidden, head, dropout=0.0):
        super(Model, self).__init__()
        self.mha_news = NewsAttentionModel(D_in, D_hidden, head, dropout)
        self.mha_keywords = KeywordAttentionModel(D_in, D_hidden, head)
        self.attention_weight = torch.nn.Parameter(torch.ones(1))  # 学习融合权重
        self.layernorm = torch.nn.LayerNorm(D_hidden)

    def forward(self, keywords, news_content, mask=None):
        # news_embd = self.mha_news(news_content, mask)
        # kw_embd = self.mha_keywords(keywords, news_content, mask)
        # # 拼接
        # cat_embd = torch.cat((news_embd, kw_embd), dim=2)
        # return cat_embd, 1
        news_embd = self.layernorm(news_content)
        return news_embd, 1

def indices_uniform_sampling(N, story_dict, cluster_confs, min_cluster_size=1):
    """Samples elements uniformly across labels.

    Args:
        N (int): size of returned data.
        cluster_indxs: indexes of datapoints in each cluster.
        cluster_sims: similarity scores of each datapoint in the cluster.
        min_cluster_size (int): minimum number of elements a cluster must have to be considered.

    Returns:
        list: sampled indices.
    """
    # 1. 统计满足最小簇大小的簇数量
    nmb_non_empty_clusters = sum(1 for i in story_dict.values() if len(i) >= min_cluster_size)

    if nmb_non_empty_clusters == 0:
        raise ValueError("No clusters meet the minimum size requirement.")

    size_per_cluster = int(N / nmb_non_empty_clusters) + 1
    res = np.array([])

    # 2. 对符合条件的簇进行采样
    choice_number = {}
    for story_idx, article_idx in story_dict.items():
        if len(article_idx) < min_cluster_size:
            continue
        if args.aug_flag and len(article_idx) < size_per_cluster:
            choice_number[story_idx] = size_per_cluster - len(article_idx)
        indexes = np.random.choice(
            article_idx,
            size_per_cluster,
            # replace=False,
            replace=(len(article_idx) <= size_per_cluster),
            p=softmax(cluster_confs[story_idx])  # 使用 softmax 作为采样概率分布
        )
        res = np.concatenate((res, indexes))

    # 3. 随机打乱结果并确保采样数量为 N
    np.random.shuffle(res)
    res = list(res.astype('int'))

    # 如果采样的数量超过 N，则裁剪多余部分
    if len(res) >= N:
        return res[:N], False, choice_number

    # 如果采样数量不足，返回数据增强的True
    res += res[: (N - len(res))]  # 循环补齐
    return res, False, choice_number

def aug_samples(window, story_dict, choice_number):
    aug_tensors = []
    aug_masks = []
    aug_class_indices = []

    for story_index, sample_num in choice_number.items():
        sample_indexs = np.random.choice(story_dict[story_index], sample_num, replace=True)
        for sample_index in sample_indexs:
            # 1. 通过 LLM 生成新的增强句子
            refer_article = window.loc[sample_index, 'summary']
            prompts = get_llm_prompts('samples_generate', refer_article)
            llm_generated_article = get_llm_output(prompts)
            # llm_generated_article = "afdsafa sdfasdf"
            # 2. 生成新的张量, 进行掩码
            new_tensor = embedding(llm_generated_article)
            new_tensor = torch.tensor(new_tensor)
            new_tensor = new_tensor.unsqueeze(0).expand(1, -1).cuda()
            new_mask = masking(new_tensor)
            # 4. 记录增强样本
            aug_tensors.append(new_tensor)
            aug_masks.append(new_mask)
            aug_class_indices.append(story_index)

    aug_tensors = torch.stack(aug_tensors)
    aug_masks = torch.stack(aug_masks)

    return aug_tensors, aug_masks, aug_class_indices


def calculate_diversity(window):
    """
    计算窗口中的主题多样化程度。
    """
    # 使用基尼系数（Gini Coefficient）作为衡量标准。

    topics = window['discovered_story'].values  # 获取窗口内所有故事的类别
    topic_counts = Counter(topics)
    total = sum(topic_counts.values())

    # 计算每个类别的占比
    proportions = [count / total for count in topic_counts.values()]
    proportions = np.array(proportions)

    # 归一化比例
    proportions = proportions / np.sum(proportions)

    # 计算基尼系数
    n = len(proportions)
    diff_matrix = np.abs(proportions[:, np.newaxis] - proportions[np.newaxis, :])
    gini = np.sum(diff_matrix) / (2 * n)

    return gini

    # 使用基尼不纯度（Gini Impurity）作为衡量标准。
    # topics = window['discovered_story'].values
    # topic_counts = Counter(topics)
    # total = sum(topic_counts.values())
    #
    # proportions = [count / total for count in topic_counts.values()]
    # proportions = np.array(proportions)
    # proportions = proportions / np.sum(proportions)
    # gini = 1 - np.sum(proportions ** 2)
    # return gini


def comp_window_size(window, df_org, window_end_date, threshold=0.75, move_step=1, max_size=3):
    """
    根据多样化程度动态调整窗口大小。

    :param window: 当前窗口数据（Pandas DataFrame）。
    :param df_org: 原始数据集（Pandas DataFrame）。
    :param window_end_date: 当前窗口结束日期（Pandas Timestamp）。
    :param move_step: 向前扩展窗口的步长（以天为单位）。
    :param threshold: 多样化程度阈值。
    :param max_size: 最大窗口大小。
    :return: (调整后的窗口数据(Pandas DataFrame)，调整后的起始日期(Pandas Timestamp))。
    """
    # 判断 max_size 是否需要调整
    min_allowed_date = pd.Timestamp('2014-01-01')  # 设置最早日期
    # if (window_from_date - pd.DateOffset(days=max_size)) < min_allowed_date:
    #     max_size = (window_from_date - min_allowed_date).days

    current_window = window.copy()
    current_start_date = window['date'].min()

    # 获取日期范围,计算时间跨度(天数)
    time_span = (current_window['date'].max() - current_window['date'].min()).days
    while time_span < max_size:
        # 计算当前窗口的多样化程度
        diversity = calculate_diversity(current_window)
        # if diversity >= threshold:  # gini不纯度 越大，多样化越高 （最好结果是0.5）
        if diversity <= threshold:  # gini Coefficient 越大，多样化越低
            print(f"diversity:{diversity}")
            break

        # 向前扩展窗口
        expand_start_date = current_start_date - pd.DateOffset(days=move_step)

        # 确保扩展不会超出边界
        if expand_start_date <= min_allowed_date:
            break

        # 合并扩展数据
        current_window = df_org[
            (df_org['date'] >= expand_start_date) &
            (df_org['date'] <= window_end_date)
        ]
        current_start_date = expand_start_date  # 更新当前窗口的起始日期

        time_span = (current_window['date'].max() - current_window['date'].min()).days
        if time_span >= max_size:
            break

    # 获取日期范围,计算时间跨度（天数）
    a, b = current_window['date'].min(), current_window['date'].max()
    print(a, b, (b - a).days)
    return current_window




if __name__ == '__main__':
    args = Args.args
    print("Parameters parsed:", args)
    new_list = []
    old_list = []
    di = set()
    if args.LLM_mode == 'llama3.1-8b' or args.LLM_mode == 'llama3.2-3b':
        model_path = '/data/zhangruwen/premodel/llama3.1-8b'
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.float16,
        #     device_map="auto",  # 自动分配多卡
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        # )
        pipe = pipeline("text-generation", model=model_path,
            device=3, torch_dtype=torch.float16)
    elif args.LLM_mode == 'deepseek-qwen':
        pipe = pipeline("text-generation", model='/data/zhangruwen/premodel/deepseek-r1-distill-qwen-7B',
                        device=3, torch_dtype=torch.bfloat16)
    elif args.LLM_mode == 'deepseek-llama':
        pipe = pipeline("text-generation", model='/data/zhangruwen/premodel/deepseek-r1-distill-llama-8B',
                        device=3, torch_dtype=torch.bfloat16)
    else:
        pipe = None

    # Load GPU
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device --', torch.cuda.current_device(), torch.cuda.get_device_name(args.GPU_NUM))

    # Load dataset and initial sentence representations/masks
    print("Loading Dataset....")
    df_org = pd.read_json('/data/zhangruwen/Stroy/dataset/News14/News14_step2_summary_kw444.json')
    masked_tensors = torch.load('/data/zhangruwen/Stroy/dataset/News14/News14_masked_embds_sample_summary.pt').cuda()
    masks = torch.load('/data/zhangruwen/Stroy/dataset/News14/News14_masks_sample_summary.pt').cuda()

    '''
        Model initialize
    '''
    print("Model initializing...")
    D_in = masked_tensors[0].shape[1]  # input dimension
    D_hidden = D_in  # output dimension
    model = Model(D_in, D_hidden, args.head, args.dropout).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    df_org['mean_cluster'] = -1
    df_org['discovered_story'] = -1  # cluster initialize
    df_org['story_conf'] = -1  # confidence initialize
    for k, v in df_org['story'].items():
        if int(k) < 63:
            di.add(v)

    '''
        Initialzie story with the first window
    '''
    window = df_org[(df_org['date'] < args.begin_date)]  # first window

    '''采用表征keywords'''
    each_win_keywords = [' '.join(sublist) for sublist in df_org['keywords'][window.index]]
    each_win_keywords_emb = embedding(each_win_keywords)
    keywords_embd = torch.from_numpy(each_win_keywords_emb)
    keywords_embd = keywords_embd.mean(dim=0)
    keywords_embd = keywords_embd.to(torch.float32)

    keywords_embds = keywords_embd.unsqueeze(0).expand(1, -1)
    keywords_embds = keywords_embds.unsqueeze(0).repeat(len(window.index), 1, 1).cuda()
    mean_embds = model(keywords_embds, masked_tensors[window.index], masks[window.index])[0].squeeze(1).cpu().detach().numpy()


    intinal_labels_set = list(df_org.loc[window.index, 'story'].unique())
    mapping_dict = {value: index for index, value in enumerate(intinal_labels_set)}
    intinal_labels = [mapping_dict[i] for i in df_org.loc[window.index, 'story']]

    df_org.loc[window.index, 'discovered_story'] = intinal_labels  # 给window里面的article赋值标签
    tuned_centers_0 = {label: [] for label in mapping_dict.values()}
    for i in zip(intinal_labels, mean_embds):
        tuned_centers_0[i[0]].append(i[1])
    tuned_centers = {k: sum(tuned_centers_0[k])/len(tuned_centers_0[k]) for k in tuned_centers_0.keys()}
    tuned_centers = np.stack(tuned_centers.values())

    story_confs = []
    for i in zip(mean_embds, intinal_labels):
        story_confs.append(cosine_similarity([i[0]], [tuned_centers[i[1]]])[0][0])
    df_org.loc[window.index, 'story_conf'] = story_confs

    '''
        为每个类生成对应的summary
    '''
    summary_dict = {label: [] for label in mapping_dict.values()}
    re_summary_dict = summary_dict
    # 收集每个类的全部新闻
    for i in window.index:
        summary_dict[df_org.loc[i, 'discovered_story']].append(df_org.loc[i, 'summary'])
    # 交给LLM生成summary
    print("Intinal the summary......")
    for k, v in summary_dict.items():
        prompt = get_llm_prompts("summary", v)
        llm_summary = get_llm_output(prompt, pipe)
        re_summary_dict[k] = llm_summary

    '''
        Initialzie model with the initial stories
    '''
    window = df_org.loc[window.index]
    init_epoch = args.init_epoch
    losses = []

    target_index = window[window.story_conf >= args.sample_thred].index
    sample_prob = window[window.story_conf >= args.sample_thred].story_conf.values / np.sum(
        window[window.story_conf >= args.sample_thred].story_conf.values)

    existing_tuned_centers = list(df_org.loc[window.index, 'discovered_story'].unique())
    center_list = [torch.tensor(tuned_centers[center]) for center in existing_tuned_centers]  # 将列表中的每个数组转换为 PyTorch 张量
    class_embds = torch.stack(center_list).cuda()  # 将张量堆叠成一个二维张量


    print("Begin initializing with the first window\n")

    warmup_epochs = init_epoch // 3
    num_itr = int(len(window) / args.batch) + 1

    for e in trange(init_epoch):
        current_lr = adjust_learning_rate(optimizer, e, init_epoch, args.lr, args.lr * 0.01, warmup_epochs)

        for itr in range(num_itr):
            samples = np.random.choice(target_index, args.batch, p=sample_prob)  # window.index

            keywords_embds = keywords_embd.unsqueeze(0).expand(1, -1)
            sample_keywords_embds = keywords_embds.unsqueeze(0).repeat(len(samples), 1, 1).cuda()

            sample_outputs = model(sample_keywords_embds, masked_tensors[samples], masks[samples])[0].squeeze(1)

            class_indices = [existing_tuned_centers.index(c) for c in df_org.loc[samples, 'discovered_story']]


            loss = infonce_loss(sample_outputs, class_indices, class_embds, args.temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Center update
    model.eval()
    for c in df_org.loc[window.index, 'discovered_story'].unique():
        if c < 0: continue
        cluster_idx = window[window['discovered_story'] == c].index
        keywords_embds = keywords_embd.unsqueeze(0).expand(1, -1)
        keywords_embds = keywords_embds.unsqueeze(0).repeat(len(cluster_idx), 1, 1).cuda()

        outputs = model(keywords_embds, masked_tensors[cluster_idx], masks[cluster_idx])

        # All output center
        tuned_centers[c] = outputs[0].squeeze(1).mean(axis=0).cpu().detach().numpy()

        df_org.loc[cluster_idx, 'story_conf'] = cosine_similarity(outputs[0].squeeze(1).cpu().detach().numpy(),
                                                                  tuned_centers[c].reshape(1, -1)).reshape(-1)

    losses.append(loss.item())

    '''
        Start sliding window evaluation
    '''
    losses = []
    tuned_ps, tuned_rs, tuned_f1s, tuned_amis, tuned_aris = [], [], [], [], []
    all_times, eval_times, train_times = [], [], []

    num_windows = len(df_org[(df_org['date'] >= args.begin_date)].date.unique())
    print("Begin evaluating sliding windows\n")

    for i in trange(num_windows):
        # 当前窗口的起始日期
        window_from_date = pd.to_datetime(args.begin_date) + pd.DateOffset(days=i * args.slide_size - args.window_size + 1)
        # 滑动窗口的起始日期
        slide_from_date = pd.to_datetime(args.begin_date) + pd.DateOffset(days=i * args.slide_size)
        # 当前窗口的结束日期
        to_date = pd.to_datetime(args.begin_date) + pd.DateOffset(days=(i + 1) * args.slide_size)
        slide = df_org[(df_org['date'] >= slide_from_date) & (df_org['date'] < to_date)]
        # window = df_org[(df_org['date'] >= window_from_date) & (df_org['date'] < to_date)]
        window = df_org[(df_org['date'] >= window_from_date) & (df_org['date'] <= slide_from_date)]

        # 计算动态滑动window_size大小
        new_window = comp_window_size(window, df_org, slide_from_date)
        non_overlapping = new_window.index

        # if len(non_overlapping) == 0:
        #     non_overlapping = window.index

        if len(slide) > 0:
            start_time = time.time()
            window_time = df_org['date'][non_overlapping]
            each_win_keywords = [' '.join(sublist) for sublist in df_org['keywords'][non_overlapping]]
            each_win_keywords_emb = embedding(each_win_keywords)
            each_win_keywords_emb = torch.from_numpy(each_win_keywords_emb)

            for slide_i in range(len(slide)):
                # print("slide_i:", slide.index[slide_i])
                '''1. 求关键字与聚类中心的相似度'''
                # 时间损失熵 * keywords
                slide_i_time = df_org['date'][slide.index[slide_i]]
                time_diff_in_days = abs(slide_i_time - window_time).dt.days  # 转化为以天为计量单位
                time_weight = np.exp(-args.alpha * time_diff_in_days.values)  # 时间衰减函数
                time_weight = torch.from_numpy(time_weight)
                time_weight_expanded = time_weight.unsqueeze(1)

                keywords_embd = each_win_keywords_emb * time_weight_expanded
                keywords_embd = keywords_embd.mean(dim=0)
                keywords_embd = keywords_embd.to(torch.float32)

                # Evaluating new articles
                model.eval()
                keywords_embds = keywords_embd.unsqueeze(0).expand(1, -1).cuda()
                keywords_embds = keywords_embds.unsqueeze(0).repeat(len(slide.index), 1, 1)
                outputs = model(keywords_embds, masked_tensors[slide.index], masks[slide.index])

                tuned_embds = outputs[0].squeeze(1).cpu().detach().numpy()
                existing_tuned_centers = [int(c) for c in df_org.loc[window.index, 'discovered_story'].unique() if c != -1]

                '''2. 判断是否是新的类'''
                if len(existing_tuned_centers) > 0:
                    sim = cosine_similarity([tuned_embds[slide_i]], tuned_centers[existing_tuned_centers])[0]
                    max_sim = np.max(sim)
                    max_sim_index = existing_tuned_centers[np.argmax(sim)]
                else:
                    sim = [-1]
                    max_sim = np.max(sim)
                    max_sim_index = -1

                # 统计数据的先验分布
                if df_org['story'][slide.index[slide_i]] not in di:
                    new_list.append(float(max_sim))
                    di.add(df_org['story'][slide.index[slide_i]])
                else:
                    old_list.append(float(max_sim))

                article = df_org['summary'][slide.index[slide_i]]
                flag, re_summary_dict = judge_class(max_sim, max_sim_index, re_summary_dict, article, tuned_centers, slide.index[slide_i], pipe)
                if flag != -1:
                    df_org.loc[slide.index[slide_i], 'discovered_story'] = flag
                    df_org.loc[slide.index[slide_i], 'story_conf'] = max_sim
                else:
                    df_org.loc[slide.index[slide_i], 'discovered_story'] = len(tuned_centers)
                    df_org.loc[slide.index[slide_i], 'story_conf'] = 1
                    existing_tuned_centers.append(len(tuned_centers))
                    tuned_centers = np.vstack((tuned_centers, tuned_embds[slide_i]))

            # Update intermediate evaluation metrics
            if args.true_story:
                eval_results = eval_metric(df_org.loc[window.index, 'story'], df_org.loc[
                    window.index, 'discovered_story'])  # precision, recall, fscore, ami, ari
                tuned_ps.append(np.round(eval_results[0], 4))
                tuned_rs.append(np.round(eval_results[1], 4))
                tuned_f1s.append(np.round(eval_results[2], 4))
                tuned_amis.append(np.round(eval_results[3], 4))
                tuned_aris.append(np.round(eval_results[4], 4))
                '''new'''
                print("Dataset", "begin_date", "B3-P", "B3-R", "B3-F1", "AMI", "ARI")
                print(args.dataset, args.begin_date, ":", np.round(np.mean(tuned_ps), 4),
                      np.round(np.mean(tuned_rs), 4), np.round(np.mean(tuned_f1s), 4),
                      np.round(np.mean(tuned_amis), 4), np.round(np.mean(tuned_aris), 4))

            eval_times.append(time.time() - start_time)

            # Updating model
            window = df_org.loc[window.index]
            slide = df_org.loc[slide.index]

            model.train()

            num_itr = int(len(window) / args.batch) + 1

            existing_tuned_centers = list(window.discovered_story.unique())  # target stories
            class_embds = torch.tensor(tuned_centers[existing_tuned_centers]).cuda()

            target_index = window[window.story_conf >= args.sample_thred].index
            sample_prob = window[window.story_conf >= args.sample_thred].story_conf.values / np.sum(
                window[window.story_conf >= args.sample_thred].story_conf.values)

            # story_dict：每个story包含的文章索引article_idx
            # cluster_sims：每个story中文章的相似度
            story_dict = window[window['discovered_story'] != -1].groupby('discovered_story')['id'].apply(list).to_dict()
            cluster_confs = {story_idx: window.loc[article_ids, 'story_conf'].values for story_idx, article_ids in story_dict.items()}

            warmup_epochs = args.epoch // 3
            for e in range(args.epoch):
                current_lr = adjust_learning_rate(optimizer, e, args.epoch, current_lr, current_lr * 0.01, warmup_epochs)

                for itr in range(num_itr):
                    samples, len_flag, choice_number = indices_uniform_sampling(args.batch, story_dict, cluster_confs)
                    keywords_embds = keywords_embd.unsqueeze(0).expand(1, -1)
                    sample_keywords_embds = keywords_embds.unsqueeze(0).repeat(len(samples), 1, 1).cuda()

                    sample_outputs = model(sample_keywords_embds, masked_tensors[samples], masks[samples])[0].squeeze(1)

                    class_indices = [existing_tuned_centers.index(c) for c in window.loc[samples, 'discovered_story']]

                    if args.aug_flag and len_flag:
                        aug_tensors, aug_masks, aug_class_indices = aug_samples(window, story_dict, choice_number)
                        aug_tensors, aug_masks = aug_tensors.cuda(), aug_masks.cuda()
                        aug_keywords_embds = keywords_embds.unsqueeze(0).repeat(aug_tensors.shape[0], 1, 1).cuda()
                        aug_sample_outputs = model(aug_keywords_embds, aug_tensors, aug_masks)[0].squeeze(1)
                        sample_outputs = torch.concat((sample_outputs, aug_sample_outputs))
                        class_indices = class_indices + aug_class_indices

                    loss = infonce_loss(sample_outputs, class_indices, class_embds, args.temp)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            losses.append(loss.item())

            '''
                Updating story representations
            '''
            torch.save(model.state_dict(), args.save_model)
            model.eval()
            for c in df_org.loc[window.index, 'discovered_story'].unique():
                if c < 0: continue
                cluster_idx = window[window['discovered_story'] == c].index
                keywords_embds = keywords_embd.unsqueeze(0).repeat(len(cluster_idx), 1, 1).cuda()

                outputs = model(keywords_embds, masked_tensors[cluster_idx], masks[cluster_idx])

                tuned_centers[c] = outputs[0].squeeze(1).mean(axis=0).cpu().detach().numpy()

                df_org.loc[cluster_idx, 'story_conf'] = cosine_similarity(outputs[0].squeeze(1).cpu().detach().numpy(),
                                                                          tuned_centers[c].reshape(1, -1)).reshape(-1)

            train_times.append(time.time() - start_time - eval_times[-1])
            all_times.append(time.time() - start_time)

    '''
        Report final evaluation metrics
    '''
    df_org[['id', 'date', 'title', 'story', 'discovered_story']].to_json(args.output_result)
    print("Total " + str(
        sum(df_org.discovered_story.value_counts() > args.min_articles)) + " valid stories are found. The output is saved to output.json")
    if args.true_story:
        print("Dataset", "begin_date", "B3-P", "B3-R", "B3-F1", "AMI", "ARI", "all_time", "eval_time", "train_time")
        print(args.dataset, args.begin_date, ":",
              np.round(np.mean(tuned_ps), 4),
              np.round(np.mean(tuned_rs), 4),
              np.round(np.mean(tuned_f1s), 4),
              np.round(np.mean(tuned_amis), 4),
              np.round(np.mean(tuned_aris), 4),
              np.round(np.mean(all_times), 4),
              np.round(np.mean(eval_times), 4),
              np.round(np.mean(train_times), 4))
    else:
        print("Dataset", "begin_date", "all_time", "eval_time", "train_time")
        print(args.dataset, args.begin_date, ":",
              np.round(np.mean(all_times), 4),
              np.round(np.mean(eval_times), 4),
              np.round(np.mean(train_times), 4))

    # 写re_summary_dict
    with open(args.llm_summary_dict, "w") as json_file:
        json.dump(re_summary_dict, json_file, indent=4)
    # print("sim_list", sim_list)
    # print("diff_list", diff_list)
    # sim_dict = {"new": new_list, "old": old_list}
    # with open("./visa/conf/llama_sim_dict.json", "w") as json_file:
    #     json.dump(sim_dict, json_file, indent=4)