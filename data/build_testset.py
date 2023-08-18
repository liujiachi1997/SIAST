"""
Author:
Date: 2021-09-09 14:56
Desc: 构建平衡测试集
"""

import os
import numpy as np


def sample_data(data_path, mode="coarse", sample_count=10, random_seed=1):
    """ 按照sample_type构建测试集"""
    samples = []
    seq_in_file = os.path.join(data_path, "seq.in")
    seq_out_file = os.path.join(data_path, "seq.out")
    import numpy as np
    np.random.seed(random_seed)
    with open(seq_in_file, "r") as f:
        seq_in_lines = f.readlines()
    with open(seq_out_file, "r") as f:
        seq_out_lines = f.readlines()
    assert len(seq_in_lines) == len(seq_out_lines)
    for seq_in_line, seq_out_line in zip(seq_in_lines, seq_out_lines):
        tokens = seq_in_line.strip().split(" ")
        tokens = [token for token in tokens if token]
        slots = seq_out_line.strip().split(" ")
        slots = [slot for slot in slots if slot]
        assert len(tokens) == len(slots)
        samples.append((tokens, slots))
    
    # print(len(samples))
    def parse_bio(tags):
        """
        给定BIO标注列表，返回一个list，包含这句话中的slot label及其对应长度。
        这里slot_label指：标签类别，不带前缀。如slot=I-round_trip, 则slot_label=round_trip.
        """
        result = []
        token_count = 0   # 用来记录除O以外的这个slot_label的槽位数，比如slots=[B-round_trip I-round_trip],则slot_label=round_trip, token_count=2.表示从此位起，有2个连续token(包含此位)都属于这次slot_label.
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
                    current_label = current_label = None
                    token_count = 0
                elif tag[0] == "":
                    token_count += 1
                else:
                    result.append((current_label, token_count))
                    current_label = tag[2:]
                    token_count = 1
            last_tag = tag
        if current_label is not None:
            result.append((current_label, token_count))
        return result

    all_slots = []   # 不带BIO前缀的标签集合
    slot2example = {}   # 每个标签的样本字典，key为标签，如round_trip，value为包含这个slot label的utterances[(tokens, slots),...]

    #***********************************************************************************
    for tokens, slots in samples:
        for slot_label, count in parse_bio(slots):
            if slot_label not in all_slots:
                all_slots.append(slot_label)
                slot2example[slot_label] = [(tokens, slots)]
            else:
                slot2example[slot_label].append((tokens, slots))
    sampled_data = []

    if mode == "coarse":
        for slot in all_slots:
            # sample_for_labeled = np.random.choice(len(slot2example[slot]), size=sample_count, replace=False)
            np.random.shuffle(slot2example[slot])
            # sample_by_slot = list(np.random.choice(slot2example[slot], size=sample_count, replace=False))
            num = 0
            for s in slot2example[slot]:
                if s not in sampled_data:
                    if num < sample_count:
                        sampled_data.append(s)
                        num += 1
                    else:
                        break
                    # elif s not in unsampled_data:
                        # unsampled_data.append(s)
    return sampled_data


if __name__ == "__main__":
    # 参数设定
    data_path = "/mnt/2/ljc/workspace/self-training-for-slot-filling/data/MITMovie-trivia/test"
    new_data_path = "/mnt/2/ljc/workspace/self-training-for-slot-filling/data/MITMovie-trivia/test"
    if not os.path.exists(new_data_path):
        os.mkdir(new_data_path)
    sample_mode = "coarse"
    sample_count = 47
    random_seed = 10
    # 抽样
    sampled_data = sample_data(data_path, sample_mode, sample_count=sample_count, random_seed=random_seed)
    # 写文件(会覆盖之前的testset)
    with open(os.path.join(new_data_path, "seq.in"), 'w', encoding='utf8')as seqin:
        with open(os.path.join(new_data_path, "seq.out"), 'w', encoding='utf8')as seqout:
            for tokens, slots in sampled_data:
                seqin.write(' '.join(tokens) + '\n')
                seqout.write(' '.join(slots) + '\n')