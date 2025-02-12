from tqdm import tqdm
import json

'''
因为在前2步的时候可能date日期的格式发生变动，所以重新处理一下json文件。
Since the format of the date date may have changed during the previous two steps, rework the json file.
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
    DATASET_NAME = 'News14'  # News14  WCEP19

    reprocess_json()
