import warnings
import logging
from pathlib import Path

import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.autograd as autograd
import flair.nn
import torch

from flair.data import Dictionary, Sentence, Token, Label
from flair.datasets import DataLoader
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path

from typing import List, Tuple, Union

from flair.training_utils import Metric, Result, store_embeddings
from .biaffine_attention import BiaffineAttention

from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import pdb
import copy

from ..variational_inference import MFVI
import time
from .sequence_tagger_model import FastSequenceTagger

class ParallelSequenceTagger(FastsequenceTagger):
    def _init_model_with_state_dict(state):
        use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if not "use_word_dropout" in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if not "use_locked_dropout" in state.keys()
            else state["use_locked_dropout"]
        )
        train_initial_hidden_state = (
            False
            if not "train_initial_hidden_state" in state.keys()
            else state["train_initial_hidden_state"]
        )
        if 'biaf_attention' in state:
            biaf_attention = state['biaf_attention']
        else:
            biaf_attention = False
        if 'token_level_attention' in state:
            token_level_attention = state['token_level_attention']
        else:
            token_level_attention = False
        if 'teacher_hidden' in state:
            teacher_hidden = state['teacher_hidden']
        else:
            teacher_hidden = 256
        use_cnn=state["use_cnn"] if 'use_cnn' in state else False
        if 'num_teachers' in state:
            num_teachers = state['num_teachers']
            model = ParallelSequenceTagger(
                hidden_size=state["hidden_size"],
                embeddings=state["embeddings"],
                tag_dictionary=state["tag_dictionary"],
                tag_type=state["tag_type"],
                use_crf=state["use_crf"],
                use_mfvi=state["use_mfvi"],
                use_rnn=state["use_rnn"],
                use_cnn=use_cnn,
                rnn_layers=state["rnn_layers"],
                dropout=use_dropout,
                word_dropout=use_word_dropout,
                locked_dropout=use_locked_dropout,
                train_initial_hidden_state=train_initial_hidden_state,
                biaf_attention=biaf_attention,
                token_level_attention=token_level_attention,
                teacher_hidden=teacher_hidden,
                num_teachers=num_teachers,
                config=state['config'],
            )
        else:

            model = ParallelSequenceTagger(
                hidden_size=state["hidden_size"],
                embeddings=state["embeddings"],
                tag_dictionary=state["tag_dictionary"],
                tag_type=state["tag_type"],
                use_crf=state["use_crf"],
                use_mfvi=state["use_mfvi"],
                use_rnn=state["use_rnn"],
                use_cnn=use_cnn,
                rnn_layers=state["rnn_layers"],
                dropout=use_dropout,
                word_dropout=use_word_dropout,
                locked_dropout=use_locked_dropout,
                train_initial_hidden_state=train_initial_hidden_state,
                biaf_attention=biaf_attention,
                teacher_hidden=teacher_hidden,
                config=state['config'],
            )
        # pdb.set_trace()
        model.load_state_dict(state["state_dict"])
        return model

    def forward_loss(
        self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        features = self.forward(data_points)
        # lengths = [len(sentence.tokens) for sentence in data_points]
        # longest_token_sequence_in_batch: int = max(lengths)

        # max_len = features.shape[1]
        # mask=self.sequence_mask(torch.tensor(lengths), max_len).cuda().type_as(features)
        loss = self._calculate_loss(features, data_points, self.mask)
        return loss

    def forward_preprocess(self, data_points: Union[List[Sentence], Sentence]):

    def distillation_forward_preprocess(self, data_points: Union[List[Sentence], Sentence], teacher_data_points: Union[List[Sentence], Sentence]=None, teacher=None, train_with_professor=False):
    	features = self.forward_preprocess(data_points)
    	
    def simple_forward_distillation_loss(
        self, data_points: Union[List[Sentence], Sentence], teacher_data_points: Union[List[Sentence], Sentence]=None, teacher=None, sort=True,
        interpolation=0.5, train_with_professor=False, professor_interpolation=0.5,
    ) -> torch.tensor:
        features = self.forward(data_points)
        lengths = [len(sentence.tokens) for sentence in data_points]
        max_len = features.shape[1]
        mask=self.mask
        if self.distill_posterior:
            # pdb.set_trace()
            # student forward-backward score
            forward_var = self._forward_alg(features, lengths, distill_mode=True)
            backward_var = self._backward_alg(features, lengths)
            # forward_var = self.forward_var
            forward_backward_score = (forward_var + backward_var) * mask.unsqueeze(-1)
            # forward_backward_score = forward_backward_score.unsqueeze(-2)
            # teacher forward-backward score
            teacher_scores = torch.stack([sentence.get_teacher_posteriors() for sentence in data_points],0)
            posterior_loss = 0
            # pdb.set_trace()
            for i in range(teacher_scores.shape[-2]):
                posterior_loss += self._calculate_distillation_loss(forward_backward_score, teacher_scores[:,:,i], mask)
            posterior_loss/=teacher_scores.shape[-2]
        else:
            posterior_loss = 0
        distillation_loss = 0
        if self.distill_crf:
            # [batch, length, kbest]
            teacher_tags=torch.stack([sentence.get_teacher_target() for sentence in data_points],0)
            # proprocess, convert k best to batch wise
            
            seq_len=teacher_tags.shape[1]
            best_k=teacher_tags.shape[-1]
            num_tags=features.shape[-1]
            # batch*best_k, seq_len
            tags=teacher_tags.transpose(1,2).reshape(-1,seq_len)
            # batch*best_k, seq_len, target_size
            features_input=features.unsqueeze(-1).repeat(1,1,1,best_k)
            features_input=features_input.permute(0,3,1,2).reshape(-1,seq_len,num_tags)
            mask_input=mask.unsqueeze(-1).repeat(1,1,best_k)
            mask_input=mask_input.transpose(1,2).reshape(-1,seq_len)
            kbatch=features_input.shape[0]

            lengths_input=torch.tensor(lengths)
            lengths_input=lengths_input.unsqueeze(-1).repeat(1,best_k)
            lengths_input=lengths_input.reshape(-1).cuda()
            # batch*bestk, seq_len, target_size -> batch*bestk, seq_len, target_size, target_size
            feature_scores=features_input.unsqueeze(-2)
            # crf_scores = feature_scores + self.transitions.view(1, 1, self.tagset_size, self.tagset_size)
            
            # pdb.set_trace()
            # features_input = torch.rand_like(features_input).cuda()
            forward_score = self._forward_alg(features_input, lengths_input)
            gold_score = self._score_sentence(features_input, tags, lengths_input, mask_input)
            distillation_loss=forward_score-gold_score
            # pdb.set_trace()
            if self.crf_attention:
                teacher_atts=torch.stack([sentence.get_teacher_weights() for sentence in data_points],0)
                att_nums=sum([len(sentence._teacher_weights) for sentence in data_points])
                
                if self.distill_with_gold:
                    # [batch, length]
                    # pdb.set_trace()
                    tag_list=torch.stack([getattr(sentence,self.tag_type+'_tags').to(flair.device) for sentence in data_points],0).long()
                    comparison=((teacher_tags-tag_list.unsqueeze(-1))!=0).float()*mask.unsqueeze(-1)
                    num_error=comparison.sum(1)
                    if self.exp_score:
                        # pdb.set_trace()
                        score_weights=torch.exp(-num_error/self.gold_const)
                    else:
                        score_error=num_error+self.gold_const
                        # [batch, best_k]
                        score_weights=self.gold_const/score_error
                    
                    teacher_atts = teacher_atts * score_weights
                    # note that the model with multiple teachers for single language is not good
                    teacher_atts = teacher_atts/teacher_atts.sum(-1,keepdim=True) * (att_nums/len(score_weights))
                #batch, kbest -> batch * kbest
                teacher_atts=teacher_atts.reshape(-1)
                distillation_loss=(distillation_loss*teacher_atts).sum()/att_nums
            else:
                distillation_loss=distillation_loss.mean()
            # distillation_loss, partition, tg_energy=self.crf_loss(crf_scores.transpose(1,0),tags.transpose(1,0),mask_input.transpose(1,0).bool())
            
        if not self.use_crf or self.distill_emission:
            if teacher is not None:
                with torch.no_grad():
                    teacher_features = teacher.forward(teacher_data_points)
            else:
                if train_with_professor and not self.biaf_attention:
                    teacher_features = torch.stack([sentence.get_professor_teacher_prediction(professor_interpolation=professor_interpolation) for sentence in data_points],0)
                elif self.biaf_attention:
                    teacher_sentfeats = torch.stack([sentence.get_teacher_sentfeats() for sentence in data_points],0)
                    # pdb.set_trace()
                    if self.token_level_attention:
                        try:
                            teacher_attention = self.biaffine(self.sent_feats.view(-1,self.sent_feats.shape[-1]),teacher_sentfeats.view(-1,teacher_sentfeats.shape[-2],teacher_sentfeats.shape[-1]))
                        except:
                            pdb.set_trace()
                        teacher_attention = teacher_attention.view(len(features),max_len,-1)
                        teacher_features = torch.stack([sentence.get_teacher_prediction(pooling='token_weighted', weight=teacher_attention[idx]) for idx,sentence in enumerate(data_points)],0)
                    else:    
                        teacher_attention = self.biaffine(self.sent_feats,teacher_sentfeats)
                        teacher_features = torch.stack([sentence.get_teacher_prediction(pooling='weighted', weight=teacher_attention[idx]) for idx,sentence in enumerate(data_points)],0)
                    # pdb.set_trace()
                    
                    # teacher_attention.expand()
                else:
                    teacher_features = torch.stack([sentence.get_teacher_prediction() for sentence in data_points],0)
                # pdb.set_trace()
            distillation_loss = self._calculate_distillation_loss(features, teacher_features, mask, teacher_is_score=not self.distill_prob)
        target_loss = self._calculate_loss(features, data_points, mask)
        # target_loss2 = super()._calculate_loss(features,data_points)
        # distillation_loss2 = super()._calculate_distillation_loss(features, teacher_features,torch.tensor(lengths))
        # pdb.set_trace()
        return interpolation * (posterior_loss + distillation_loss) + (1-interpolation) * target_loss
    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))
    def _calculate_distillation_loss(self, features, teacher_features, mask, T = 1, teacher_is_score=True):
        # TODO: time with mask, and whether this should do softmax
        if teacher_is_score:
            teacher_prob=F.softmax(teacher_features/T, dim=-1)
        else:
            teacher_prob=teacher_features
        KD_loss = torch.nn.functional.kl_div(F.log_softmax(features/T, dim=-1), teacher_prob,reduction='none') * mask.unsqueeze(-1)

        # KD_loss = KD_loss.sum()/mask.sum()
        if self.sentence_level_loss or self.use_crf:
            KD_loss = KD_loss.sum()/KD_loss.shape[0]
        else:
            KD_loss = KD_loss.sum()/mask.sum()
        return KD_loss
        # return torch.nn.functional.MSELoss(features, teacher_features, reduction='mean')
    def _calculate_loss(
        self, features: torch.tensor, sentences: List[Sentence], mask: torch.tensor,
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        try:
            tag_list=torch.stack([getattr(sentence,self.tag_type+'_tags').to(flair.device) for sentence in sentences],0).long()
        except:
            tag_list: List = []
            for s_id, sentence in enumerate(sentences):
                # get the tags in this sentence
                tag_idx: List[int] = [
                    self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                    for token in sentence
                ]
                # add tags as tensor
                tag = torch.tensor(tag_idx, device=flair.device)
                tag_list.append(tag)

            tag_list, _ = pad_tensors(tag_list)
        
        if self.use_crf:
            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tag_list, torch.tensor(lengths), mask=mask)
            score = forward_score - gold_score
            score = score.mean()
        # elif self.use_mfvi:
        #     token_feats=self.sent_feats
        #     unary_score=features
        #     # pdb.set_trace()
        #     q_value=self.mfvi(token_feats,unary_score,mask)
        #     score = torch.nn.functional.cross_entropy(q_value.view(-1,q_value.shape[-1]), tag_list.view(-1,), reduction='none') * mask.view(-1,)
        #     if self.sentence_level_loss:
        #         score = score.sum()/features.shape[0]
        #     else:
        #         score = score.sum()/mask.sum()
        else:
            score = torch.nn.functional.cross_entropy(features.view(-1,features.shape[-1]), tag_list.view(-1,), reduction='none') * mask.view(-1,)
            if self.sentence_level_loss or self.use_crf:
                score = score.sum()/features.shape[0]
            else:
                score = score.sum()/mask.sum()
        if self.posterior_constraint:
            # student forward-backward score
            # pdb.set_trace()
            forward_var = self._forward_alg(features, lengths, distill_mode=True)
            backward_var = self._backward_alg(features, lengths)
            # forward_var = self.forward_var
            forward_backward_score = (forward_var + backward_var) * mask.unsqueeze(-1)
            # fwbw_probability = F.softmax(forward_backward_score,dim=-1)
            posterior_score = torch.nn.functional.cross_entropy(forward_backward_score.view(-1,forward_backward_score.shape[-1]), tag_list.view(-1,), reduction='none') * mask.view(-1,)
            if self.sentence_level_loss:
                posterior_score = posterior_score.sum()/features.shape[0]
            else:
                posterior_score = posterior_score.sum()/mask.sum()
            
            # pdb.set_trace()
            score = (1-self.posterior_interpolation) * score + self.posterior_interpolation * posterior_score
        return score
    def _score_sentence(self, feats, tags, lens_,mask=None):
        start = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(START_TAG)], device=flair.device
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=flair.device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1).cuda()
        pad_stop_tags = torch.cat([tags, stop], 1).cuda()
        transition_mask=torch.ones(mask.shape[0],mask.shape[1]+1).type_as(mask)
        transition_mask[:,1:]=mask
        transition_mask2=torch.ones(mask.shape[0],mask.shape[1]+1).type_as(mask)
        transition_mask2[:,:-1]=mask
        transition_mask2[:,-1]=0
        pad_stop_tags = pad_stop_tags.cuda()*transition_mask2.long()+(1-transition_mask2.long())*self.tag_dictionary.get_idx_for_item(STOP_TAG)
        
        my_emission=torch.gather(feats,2,tags.unsqueeze(-1))*mask.unsqueeze(-1)
        
        my_emission=my_emission.sum(-1).sum(-1)
        # (bat_size, seq_len + 1, target_size_to, target_size_from)
        bat_size=feats.shape[0]
        seq_len=pad_stop_tags.shape[1]
        ts_energy=self.transitions.unsqueeze(0).unsqueeze(0).expand(bat_size,seq_len,self.tagset_size,self.tagset_size)

        # extract the first dimension (2nd dimension here) of transition scores
        # (bat_size, seq_len + 1, target_size_to, target_size_from) -> (bat_size, seq_len + 1, 1, target_size_from)
        ts_energy=torch.gather(ts_energy,2,pad_stop_tags.unsqueeze(-1).unsqueeze(-1).expand(bat_size,seq_len,1,feats.shape[-1]))
        # (bat_size, seq_len + 1, 1, target_size_from) -> (bat_size, seq_len + 1, target_size_from)
        ts_energy=ts_energy.squeeze(2)
        # (bat_size, seq_len + 1, target_size_from) -> (bat_size, seq_len + 1)
        ts_energy=torch.gather(ts_energy,2,pad_start_tags.unsqueeze(-1)).squeeze(-1)
        
        ts_energy=ts_energy*transition_mask
        ts_energy=ts_energy.sum(1)
        score=ts_energy+my_emission
        # my_transition=
        return score

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embeddings_storage_mode: str = "cpu",
    ) -> (Result, float):

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            metric = Metric("Evaluation")

            lines: List[str] = []
            for batch in data_loader:
                batch_no += 1

                with torch.no_grad():
                    features = self.forward(batch)
                    mask=self.mask
                    loss = self._calculate_loss(features, batch, mask)
                    tags, _ = self._obtain_labels(features, batch)

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label("predicted", tag)

                        # append both to file for evaluation
                        eval_line = "{} {} {} {}\n".format(
                            token.text,
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")
                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        (tag.tag, str(tag)) for tag in sentence.get_spans("predicted")
                    ]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                store_embeddings(batch, embeddings_storage_mode)

            eval_loss /= batch_no

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.micro_avg_f_score(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss
    # def save(self, model_file: Union[str, Path]):
    #     """
    #     Saves the current model to the provided file.
    #     :param model_file: the model file
    #     """
    #     model_state = self._get_state_dict()

    #     torch.save(model_state, str(model_file))
