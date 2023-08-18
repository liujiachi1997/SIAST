# -*- encoding: utf-8 -*-

# here put the import lib
from typing import List, Dict, Optional, Union
from allennlp.modules.token_embedders import embedding

import torch
import logging

from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_mean, masked_max

from losses.dmto import DeconfusionMarginTrainingLoss

#------------------------------------------------------------------------------------------------------------
@Model.register("slot_tagger")
class SlotTaggingModel(Model):
    def __init__(self, 
                 vocab: Vocabulary,
                 embedder: Union[TextFieldEmbedder, PretrainedTransformerMismatchedEmbedder],
                 encoder: Optional[Seq2SeqEncoder] = None,
                 dropout: Optional[float] = None,
                 use_crf: bool = True,
                 cls_num_list: List = None,
                 slot_weight: torch.Tensor = None,
                 max_m: float = 0.5,
                 s: int = 30,
                 acsl_threshold: float = 0.5,
                 lambda1: float = 1,            
                 lambda2: float = 1
                ) -> None:   
        super().__init__(vocab)
        
        if isinstance(embedder.token_embedder_tokens, PretrainedTransformerMismatchedEmbedder):
            self.user_bert = True
            self.bert_embedder = embedder
        else:
            self.user_bert = False
            self.basic_embedder = embedder
        self.encoder = encoder
        if encoder:
            hidden2tag_in_dim = encoder.get_output_dim()
        else:
            hidden2tag_in_dim = self.bert_embedder.get_output_dim()
        self.hidden2tag = TimeDistributed(torch.nn.Linear(in_features=hidden2tag_in_dim,
                                                          out_features=vocab.get_vocab_size("labels")))
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.loss_weight = lambda1
        self.loss_func = DeconfusionMarginTrainingLoss(cls_num_list, max_m=max_m, weight=slot_weight, s=s, acsl_thre=acsl_threshold, lambda_ = lambda2)


        self.use_crf = use_crf
        if use_crf:
            crf_constraints = allowed_transitions(
                constraint_type="BIO",
                labels=vocab.get_index_to_token_vocabulary("labels")
            )
            self.crf = ConditionalRandomField(
                num_tags=vocab.get_vocab_size("labels"),
                constraints=crf_constraints,
                include_start_end_transitions=True
            )
        self.f1 = SpanBasedF1Measure(vocab,
                                     tag_namespace= "labels",
                                     ignore_classes=[],
                                     label_encoding="BIO")
        self.loss_metric = {'f1': self.f1}
        

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None,
                ) -> Dict[str, torch.Tensor]:
        output = {}

        mask = get_text_field_mask(sentence)   # with 0 where the tokens are padding, and 1 otherwise. `padding_id` specifies the id of padding tokens.
        output["mask"] = mask
        if self.user_bert:
            embeddings = self.bert_embedder(sentence)
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["embeddings"] = embeddings
        else:
            embeddings = self.basic_embedder(sentence)
            if self.dropout:
                embeddings = self.dropout(embeddings)
            output["embeddings"] = embeddings
        
        if not self.training:
            output.update(self._inner_forward(embeddings, mask, labels))
        else:
            output.update(self._inner_forward(embeddings, mask, labels))

        return output


    def _inner_forward(self,
                       embeddings: torch.Tensor,
                       mask: torch.Tensor,
                       labels: torch.Tensor
                       ):
        output = {}

        if not hasattr(self.encoder._module, '_flattened'):
            self.encoder._module.flatten_parameters()
        
        if self.encoder:
            encoder_out = self.encoder(embeddings, mask)
            if self.dropout:
                encoder_out = self.dropout(encoder_out)
            output["encoder_out"] = encoder_out
        else:
            encoder_out = embeddings
        tag_logits = self.hidden2tag(encoder_out)
        output["tag_logits"] = tag_logits
        output['mask'] = mask
        if self.use_crf:
            best_paths = self.crf.viterbi_tags(tag_logits, mask)
            predicted_tags = [x for x, y in best_paths]  # get the tags and ignore the score
            predicted_score = [y for _, y in best_paths]
            output["predicted_tags"] = predicted_tags
            output["predicted_score"] = predicted_score
        else:
            output["predicted_tags"] = torch.argmax(tag_logits, dim=-1)
        

        if labels is not None:
            output["loss"] = 0
            loss = self.loss_func(tag_logits.view(-1, tag_logits.shape[-1]), labels.view(-1))

            output["loss"] += self.loss_weight * loss

            if self.use_crf:
                log_likelihood = self.crf(tag_logits, labels, mask)  # returns log-likelihood  tag_logits: torch.Size([64, 23, 72])
                
                output["loss"] += -1.0 * log_likelihood  # add negative log-likelihood as loss

                # Represent viterbi tags as "class probabilities" that we can feed into the metrics
                class_probabilities = tag_logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
                self.f1(class_probabilities, labels, mask.to(dtype=torch.bool))
            else:
                output["loss"] += sequence_cross_entropy_with_logits(tag_logits, labels, mask)
                self.f1(tag_logits, labels, mask.to(dtype=torch.bool))

        return output
        

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric = self.f1.get_metric(reset)
        results = {
            "precision": metric["precision-overall"],
            "recall": metric["recall-overall"],
            "f1": metric["f1-measure-overall"],
        }
        return results
