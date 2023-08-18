#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
# here put the import lib
from typing import Tuple, List
from overrides import overrides
import json
import math

from allennlp.common.util import JsonDict
from allennlp.data import Instance, instance
from allennlp.predictors import Predictor


@Predictor.register("slot_tagging_predictor")
class SlotTaggingPredictor(Predictor):
    def predict(self, inputs: List[JsonDict]) -> JsonDict:

        instances = self._batch_json_to_instances(inputs)
        output_dict = self.predict_batch_instance(instances)
        assert len(output_dict) == len(inputs)
        outputs = []
        for i, output in enumerate(output_dict):
            _output = {
                "tokens": inputs[i]["tokens"],
                "predict_labels": [self._model.vocab.get_token_from_index(index, namespace="labels")
                                for index in output["predicted_tags"]],
                "predict_score": output["predicted_score"],
                "tag_logits": output["tag_logits"],
                "mask": output["mask"]
            }
            
            if "true_labels" in inputs[0].keys():
                _output["true_labels"] = inputs[i]["true_labels"]
            outputs.append(_output)
        return outputs


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = ' '.join(json_dict["tokens"])
        instance = self._dataset_reader.text_to_instance(text=tokens)
        return instance


def stat_slot_nums(data_path, mode="all"):
    """ 统计槽位数量.
    
    Args:
        path: 数据路径
        mode: choices=["all", "add"], default="all"; "all": 本轮迭代使用的所有训练数据; "add": 本轮训练数据中 从上一轮添加进来的伪标注数据.

    Returns:
        labeled_slots_counter: `dict`, key: `str`, slot label name, value: `int`, num.
    """

    def get_all_samples(data_path):
        """
        获取所有指定split的样本，返回（tokens, slots）元组。
        """
        samples = []
        if mode == "all":
            seq_in_file = os.path.join(data_path, "seq.in")
            seq_out_file = os.path.join(data_path, "seq.out")
        elif mode == "add":
            seq_in_file = os.path.join(data_path, "new_added_seq.in")
            seq_out_file = os.path.join(data_path, "new_added_seq.out")
        else:
            raise Exception("`mode` wrong! ")

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
            assert len(tokens) == len(slots), tokens
            samples.append((tokens, slots))
        return samples

    all_samples = get_all_samples(data_path)
    labeled_slots_counter = dict()
    for _, slots in all_samples:
        for slot in slots:
            if slot.startswith("B-"):
                slot_label = slot[2:]
                labeled_slots_counter[slot_label] = labeled_slots_counter.get(slot_label, 0) + 1

    return labeled_slots_counter


def calculate_slot_weight(labeled_slots_counter, rho=1, pattern_id=1, slot_label_list=None, save_data_path=None):
    """ 根据数量关系计算每个槽位的权重(slot-span-level weight)
    
        Args:
            lebeled_slots_counter： `dict`, key: `str`, slot label name, value: `int`, num.
            rho: Hyper-parameter.
            slot_label_list: 槽位标签列表
            save_data_path: 权重保存目录
            pattern_id: 对于新增槽位数目为0的处理，采用方式1还是2

        Returns:
            `dict`: key: slot label name, value: probability. 
    """
    slot_weights_dict = {}

    # ---- 防止一些情况下added数据中某些slot出现次数为0时 计算出错; 两种处理方法(二选一)----------
    if slot_label_list:
        # 方式一：
        # 若没出现，这个槽位的权重置为1
        if pattern_id == 1:
            for slot in slot_label_list:
                slot_weights_dict[slot] = 1
        elif pattern_id == 2:
            # 方式二：
            # 若没出现，将该槽位出现次数置为1
            for slot in slot_label_list:
                if slot not in labeled_slots_counter.keys():
                    labeled_slots_counter[slot] = 1

    # class_num = len(labeled_slots_counter)                                         # 有多少种槽位, 公式中的 L
    # nums_sort_list = sorted(list(labeled_slots_counter.values()), reverse=True)    # 数量降序排序
    nums_sort_list = sorted(list(set((labeled_slots_counter.values()))), reverse=True)   # 数量去重后降序排序
    L = len(nums_sort_list)
    N_1 = nums_sort_list[0]                                             # 最频繁的类别的数量
    
    for slot, num in labeled_slots_counter.items():
        num_rank = nums_sort_list.index(num) + 1 
        prob = (nums_sort_list[L-num_rank]/N_1) ** rho                    # 公式中的 rho
        slot_weights_dict[slot] = prob
    
    if save_data_path:
        # Save file
        with open(os.path.join(save_data_path, "slot_weights.json"),"w") as f:
            json.dump(slot_weights_dict, f, indent=4)
    
    return slot_weights_dict


def calculate_sent_confidence(slot_weight_dict, predict_labels, max_score, gamma=2):
    """ 计算句子置信度
    
    Args:
        slot_weight_dict: 
        predict_labels: 预测标签list.
        max_score: 预测标签的置信度(logits)得分.
        gamma: int型超参. 特例："infinity"表示当gamma为无穷大时, 相当于python当中的float("int").

    Returns:
        sent_confidence: float, 句子的置信度得分。由其中每个实体槽位加权平均获得。
    """

    score_sum, weight_sum = 0, 1e-8

    for i, p in enumerate(predict_labels):    # 返回的预测标签p直接是标签,如I-artist
        if p.startswith('B') or p.startswith('I'):
            if gamma == "infinity":
                score_sum += slot_weight_dict[p[2:]] * max_score[i]
                weight_sum += slot_weight_dict[p[2:]]
            elif isinstance(gamma, int):
                score_sum += (1 + gamma * slot_weight_dict[p[2:]]) * max_score[i]
                weight_sum += (1 + gamma * slot_weight_dict[p[2:]])

    sent_confidence = score_sum/weight_sum

    return sent_confidence


def calculate_resample_prob(sent_confidence, c_min, alpha=2, beta=1, mode="exponential"):
    """ 计算句子被采样到的概率 
    
    Args:
        mode: "exponential": 指数型, fun1; "logarithmic": 对数型,fun2; "linear": 线性, fun3.
            For "exponential", alpha>0, beta <= 1;
            For "logarithmic", alpha>0, beta >= 1;
            For "linear", alpha>0, beta >= 0;
        
    Returns:
        resample probability.
    """

    if mode == "exponential":
        return (math.exp(alpha * (sent_confidence - c_min)) - beta) / (math.exp(alpha * (1 - c_min)) - beta)

    elif mode == "logarithmic":
        return (math.log(alpha * (sent_confidence - c_min) + beta)) / (math.log(alpha * (1 - c_min) + beta))
    
    elif mode == "linear":
        return (alpha * (sent_confidence - c_min) + beta) / (alpha * (1 - c_min) + beta) 
    
    else:
        raise ValueError(" `calculate_resample_prob` mode: No such func! ")


if __name__ == "__main__":
    print(calculate_resample_prob(0.8082, 0.6, 2, 1, mode="exponential"))
    print(calculate_resample_prob(0.9, 0.6, 10, 1, mode="logarithmic"))
    print(calculate_resample_prob(0.9, 0.6, 2, 1, mode="linear"))
    