import argparse

class Args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='dataset/News14/News14', type=str, help="News14, WCEP19")
    parser.add_argument('--dataset_name', default='News14', type=str, help="News14, WCEP19")
    parser.add_argument('--begin_date', default='2014-01-02', type=str, help="the last date of the first window")
    parser.add_argument('--output_result', default='output/News14_output.json', type=str, help="News14, WCEP19")
    parser.add_argument('--save_model', default='output/News14_model.pth', type=str, help="News14, WCEP19")
    parser.add_argument('--llm_synopsis', default='output/News14_synopsis.json', type=str, help="News14, WCEP19")
    parser.add_argument('--LLM_mode', default='gpt-4o-mini', type=str, help="gpt-4o-mini, llama3.1-8b, llama3.2-3b, deepseek-llama, gemini-pro, gemma-7b")

    parser.add_argument('--window_size', default=1, type=int) 
    parser.add_argument('--slide_size', default=1, type=int)
    parser.add_argument('--thred', default=0.5, type=float, help="decide to initiate a new story or assign to the most confident story.")
    parser.add_argument('--sample_thred', default=0.5, type=float, help="the minimum confidence score to be sampled (the lower bound is thred)")
    parser.add_argument('--temp', default=0.2, type=float)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--init_epoch', default=12, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--head', default=4, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--true_story', default=True, type=bool)
    parser.add_argument('--alpha', default=0.2, type=float, help="the alpha of the decay weight")

    args = parser.parse_args()
