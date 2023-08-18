"""
Command:
    python3 data/generate_statistics.py --include_split valid test --output_filename label_statistics.json --action count_intents_and_slots
    python3 data/generate_statistics.py --output_filename slot_statistics.json --action count_slots_num --ignore_intent --data_path ./data/Few-NERD/intra_format
"""

import os
import json
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_split", type=str, nargs="+", default=["train", "valid", "test"],
                        help="the data split to include, valid only for function `count_intents_and_slots` ")
    parser.add_argument("--data_path", type=str, default="./data/Few-NERD-supervised")
    parser.add_argument("--output_filename", type=str, default="label_statistics.json",
                        help="the output filename")
    parser.add_argument("--ignore_intent", action="store_true", default=False,
                        help="whether to ignore intent statistics")
    parser.add_argument("--action", type=str, default="count_slots_num", choices=["count_intents_and_slots", "count_slots_num"])
    return parser.parse_args()


def get_all_samples(args):
    """
    获取所有指定split的样本，返回（tokens, intent, slots）的三元组。
    """
    samples = []
    data_path = args.data_path
    for split in args.include_split:
        seq_in_file = os.path.join(data_path, split, "seq.in")
        seq_out_file = os.path.join(data_path, split, "seq.out")
        label_file = os.path.join(data_path, split, "label")
        with open(seq_in_file, "r") as f:
            seq_in_lines = f.readlines()
        with open(seq_out_file, "r") as f:
            seq_out_lines = f.readlines()
        if args.ignore_intent:
            label_lines = ["NO_INTENT\n" for _ in range(len(seq_in_lines))]
        else:
            with open(label_file, "r") as f:
                label_lines = f.readlines()
        assert len(seq_in_lines) == len(seq_out_lines) == len(label_lines)
        for seq_in_line, seq_out_line, label_line in zip(seq_in_lines, seq_out_lines, label_lines):
            tokens = seq_in_line.strip().split(" ")
            tokens = [token for token in tokens if token]
            intent = label_line.strip()
            slots = seq_out_line.strip().split(" ")
            slots = [slot for slot in slots if slot]
            assert len(tokens) == len(slots), tokens
            samples.append((tokens, intent, slots))
    return samples


def parse_bio(tags):
    """
    给定BIO标注列表，返回一个list，包含这句话中的slot label及其对应长度。
    """
    result = []
    token_count = 0
    last_tag = "O"
    current_label = None
    for tag in tags:
        if last_tag == "O":
            if tag != "O":
                assert tag[0] == "B" or tag[0] == "I"
                current_label = tag[2:]
                token_count = 1
        else:
            if tag == "O":
                result.append((current_label, token_count))
                current_label = None
                token_count = 0
            elif tag[0] == "I":
                token_count += 1
            else:
                result.append((current_label, token_count))
                current_label = tag[2:]
                token_count = 1
        last_tag = tag
    if current_label is not None:
        result.append((current_label, token_count))
    return result


def count_intents_and_slots(samples):
    """
    统计给定数据集中的intent和slots数目
    """
    def generate_examples(tokens, intent, slots):
        return {
            "sentence": " ".join(tokens),
            "intent": intent,
            "slots": " ".join(slots)
        }
    counter = {
        "slots": {},
        "intents": {}
    }
    for tokens, intent, slots in samples:
        if intent not in counter["intents"]:
            counter["intents"][intent] = {
                "count": 1,
                "example": generate_examples(tokens, intent, slots)
            }
        else:
            counter["intents"][intent]["count"] += 1
        for slot_label, count in parse_bio(slots):
            if slot_label not in counter["slots"]:
                counter["slots"][slot_label] = {
                    "count": 1,
                    "intent_list": [intent],
                    "example": generate_examples(tokens, intent, slots),
                    "max_token_count": count
                }
            else:
                counter["slots"][slot_label]["count"] += 1
                if intent not in counter["slots"][slot_label]["intent_list"]:
                    counter["slots"][slot_label]["intent_list"].append(intent)
                if count > counter["slots"][slot_label]["max_token_count"]:
                    counter["slots"][slot_label]["max_token_count"] = count
                    counter["slots"][slot_label]["example"] = generate_examples(tokens, intent, slots)
    return counter


def plot_quantity_distribution(slots_counter_json, fig_title, save_path):
    """ 绘制槽位数量曲线图 """
    # 准备数据
    ## x 按数量从大到小排序
    sorted_slots_counter = sorted(slots_counter_json.items(), key=lambda item:item[1], reverse=True)   # 返回元素为tuple的list: [('O', 57288), ('object_type', 3023), ...] 
    ## 去除'O'
    sorted_slots_counter.pop(0)
    ## 获取x,y
    slot_name_list = [slot[0] for slot in sorted_slots_counter]
    slot_num_list = [slot[1] for slot in sorted_slots_counter]

    plt.figure(figsize=(20,16))
    plt.plot(list(range(len(slot_name_list))), slot_num_list)
    plt.title(fig_title, fontsize=25)
    plt.xticks(list(range(len(slot_name_list))), slot_name_list, rotation=90, fontsize=15)
    plt.yticks(fontsize=25)
    plt.xlabel("Slot", loc='center', fontsize=25)
    plt.ylabel("Quantity", loc='center', fontsize=25)
    plt.tight_layout()   # fix xlable坐标轴显示不全
    plt.legend()
    plt.savefig(save_path)


def count_slots_num(args):
    """ 统计数据集中每类slot_label出现的次数"""
    slots_counter = {
        "train": {},
        "valid": {},
        "test": {}
    }


    for split in ["train", "test", "valid"]:
        args.include_split = [split]
        print("split: ", args.include_split)
        all_samples = get_all_samples(args)
        for tokens, intent, slots in all_samples:
            
            for slot in slots:
                if slot.startswith("B-"):
                    slot_label = slot[2:]
                    slots_counter[split][slot_label] = slots_counter[split].get(slot_label, 0) + 1
                elif slot == "O":
                    slots_counter[split]["O"] = slots_counter[split].get("O", 0) + 1

    dataset = args.data_path.split('/')[-1]
    plot_quantity_distribution(slots_counter["train"], fig_title=f'{dataset} quantity distribution', save_path=f'{args.data_path}/trainset_quantity_distribution.png')
    return slots_counter


def main(args):
    if args.action == "count_intents_and_slots":
        all_samples = get_all_samples(args)
        counter = count_intents_and_slots(all_samples)
    elif args.action == "count_slots_num":
        counter = count_slots_num(args)
    else:
        raise Exception("Param error: No such function")
    output_file = os.path.join(args.data_path, args.output_filename)
    with open(output_file, "w") as f:
        json.dump(counter, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)