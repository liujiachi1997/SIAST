#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# here put the import lib
from typing import Dict, Iterable, List
import os
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField, SequenceLabelField, ArrayField, ListField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.dataset_readers import Conll2003DatasetReader


@DatasetReader.register("slot_filling")
class SlotFillingDatasetReader(DatasetReader):
    def __init__(self, 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        # super().__init__(lazy=lazy)
        super().__init__()
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="tokens"),
                                                 "token_characters": TokenCharactersIndexer(namespace="token_characters")}

 
    def text_to_instance(self, text: str, labels: List[str] = None) -> Instance:
        sentence_field = TextField(self.tokenizer.tokenize(text), self.token_indexers)
        fields = {"sentence": sentence_field}
        # textField = MetadataField(text)
        # fields['text'] = textField
        if labels:
            slot_label_field = SequenceLabelField(labels=labels, sequence_field=sentence_field, label_namespace="labels")
            fields["labels"] = slot_label_field
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterable[Instance]:
        token_file_path = os.path.join(file_path, "seq.in")
        label_file_path = os.path.join(file_path, "seq.out")
        with open(token_file_path, "r", encoding="utf-8") as f_token:
            token_lines = f_token.readlines()
        with open(label_file_path, "r", encoding="utf-8") as f_label:
            label_lines = f_label.readlines()
        assert len(token_lines) == len(label_lines)
        for token_line, label_line in zip(token_lines, label_lines):
            if not token_line.strip() or not label_line.strip():
                continue
            labels: List[str] = label_line.strip().split(" ")
            if len(labels) == 0:
                continue
            labels = [label.strip() for label in labels if label.strip()]
            yield self.text_to_instance(token_line.strip(), labels)

        # return super()._read(file_path)