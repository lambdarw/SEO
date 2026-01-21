from tqdm import tqdm
import json

'''
重新处理一遍json文件
'''
def reprocess_json():
    with open("./dataset/" + DATASET_NAME + "_step2_summary_kw.json", "r") as file:
        data = json.load(file)

    with open(DATASET_NAME + "_step1_summary.json", "r") as file:
        data222 = json.load(file)

    for k, v in tqdm(data['summary'].items()):
        # data['summary'][k] = f"{v[0]}: {v[1]}"
        data['date'][k] = data222['date'][k]

    with open(DATASET_NAME + "_step2_summary_kw2.json", "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    DATASET_NAME = ''  # News14  WCEP19

    reprocess_json()

