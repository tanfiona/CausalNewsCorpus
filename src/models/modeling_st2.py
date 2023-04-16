import torch
from torch import nn
from typing import Optional
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification
    )


class SignalDetector(nn.Module):
    def __init__(self, model_and_tokenizer_path="outs_test/signal_cls") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizer_path) 
        self.signal_detector = AutoModelForSequenceClassification.from_pretrained(model_and_tokenizer_path)
        self.signal_detector.eval()
        self.signal_detector.cuda()
    
    @torch.no_grad()
    def predict(self, text: str) -> int:
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor([input_ids]).cuda()
        outputs = self.signal_detector(input_ids)
        return outputs[0].argmax().item()


class ST2Model(nn.Module):
    def __init__(self, args):
        super(ST2Model, self).__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(args.model_name_or_path)

        self.dropout = nn.Dropout(0.2)
        self.ce_classifier = nn.Linear(self.config.hidden_size, 5)
        self.sig_classifier = nn.Linear(self.config.hidden_size, 3)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        ce_labels: Optional[torch.LongTensor] = None,
        sig_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        ce_logits = self.ce_classifier(sequence_output)
        sig_logits = self.sig_classifier(sequence_output)

        ce_loss = None
        if ce_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(ce_logits.view(-1, 5), ce_labels.view(-1))
        
        sig_loss = None
        if sig_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            sig_loss = loss_fct(sig_logits.view(-1, 3), sig_labels.view(-1))

        loss = None
        if ce_loss is not None and sig_loss is not None:
            loss = ce_loss + sig_loss
        return {
            'ce_logits': ce_logits,
            'sig_logits': sig_logits,
            'ce_loss': ce_loss,
            'sig_loss': sig_loss,
            'loss': loss,
        }

    
class ST2ModelV2(nn.Module):
    def __init__(self, args):
        super(ST2ModelV2, self).__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir='/data/cache/huggingface/models')
        self.model = AutoModel.from_pretrained(args.model_name_or_path, cache_dir='/data/cache/huggingface/models')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='/data/cache/huggingface/models')

        classifier_dropout = self.args.dropout
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 6)

        if args.mlp:
            self.classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.hidden_size, 6),
                nn.Tanh(),
                nn.Linear(6, 6),
            )

        if args.add_signal_bias:
            self.signal_phrases_layer = nn.Parameter(
                torch.normal(
                    mean=self.model.embeddings.word_embeddings.weight.data.mean(), 
                    std=self.model.embeddings.word_embeddings.weight.data.std(),
                    size=(1, self.config.hidden_size),
                )
            )
        
        if self.args.signal_classification and not self.args.pretrained_signal_detector:
            self.signal_classifier = nn.Linear(self.config.hidden_size, 2)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        signal_bias_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,     # [batch_size, 3]
        end_positions: Optional[torch.Tensor] = None,       # [batch_size, 3]
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if signal_bias_mask is not None and not self.args.signal_bias_on_top_of_lm:
            inputs_embeds = self.signal_phrases_bias(input_ids, signal_bias_mask)

            outputs = self.model(
                # input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            if self.args.model_name_or_path in ['facebook/bart-large', 'facebook/bart-base', 'facebook/bart-large-cnn']:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif self.args.model_name_or_path in ['microsoft/deberta-base']:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            else:               
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        sequence_output = outputs[0]
        if signal_bias_mask is not None and self.args.signal_bias_on_top_of_lm:
            sequence_output[signal_bias_mask == 1] += self.signal_phrases_layer

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, max_seq_length, 6]
        start_arg0_logits, end_arg0_logits, start_arg1_logits, end_arg1_logits, start_sig_logits, end_sig_logits = logits.split(1, dim=-1)
        start_arg0_logits = start_arg0_logits.squeeze(-1).contiguous()
        end_arg0_logits = end_arg0_logits.squeeze(-1).contiguous()
        start_arg1_logits = start_arg1_logits.squeeze(-1).contiguous()
        end_arg1_logits = end_arg1_logits.squeeze(-1).contiguous()
        start_sig_logits = start_sig_logits.squeeze(-1).contiguous()
        end_sig_logits = end_sig_logits.squeeze(-1).contiguous()

        # start_arg0_logits -= (1 - attention_mask) * 1e4
        # end_arg0_logits -= (1 - attention_mask) * 1e4
        # start_arg1_logits -= (1 - attention_mask) * 1e4
        # end_arg1_logits -= (1 - attention_mask) * 1e4

        # start_arg0_logits[:, 0] = -1e4
        # end_arg0_logits[:, 0] = -1e4
        # start_arg1_logits[:, 0] = -1e4
        # end_arg1_logits[:, 0] = -1e4

        signal_classification_logits = None
        if self.args.signal_classification and not self.args.pretrained_signal_detector:
            signal_classification_logits = self.signal_classifier(sequence_output[:, 0, :])
        # start_logits = start_logits.squeeze(-1).contiguous()
        # end_logits = end_logits.squeeze(-1).contiguous()

        arg0_loss = None
        arg1_loss = None
        sig_loss = None
        total_loss = None
        signal_classification_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()

            start_arg0_loss = loss_fct(start_arg0_logits, start_positions[:, 0])
            end_arg0_loss = loss_fct(end_arg0_logits, end_positions[:, 0])
            arg0_loss = (start_arg0_loss + end_arg0_loss) / 2

            start_arg1_loss = loss_fct(start_arg1_logits, start_positions[:, 1])
            end_arg1_loss = loss_fct(end_arg1_logits, end_positions[:, 1])
            arg1_loss = (start_arg1_loss + end_arg1_loss) / 2

            # sig_loss = 0.
            start_sig_loss = loss_fct(start_sig_logits, start_positions[:, 2])
            end_sig_loss = loss_fct(end_sig_logits, end_positions[:, 2])
            sig_loss = (start_sig_loss + end_sig_loss) / 2

            if sig_loss.isnan():
                sig_loss = 0.

            if self.args.signal_classification and not self.args.pretrained_signal_detector:
                signal_classification_labels = end_positions[:, 2] != -100
                signal_classification_loss = loss_fct(signal_classification_logits, signal_classification_labels.long())
                total_loss = (arg0_loss + arg1_loss + sig_loss + signal_classification_loss) / 4
            else:
                total_loss = (arg0_loss + arg1_loss + sig_loss) / 3
                

        return {
            'start_arg0_logits': start_arg0_logits,
            'end_arg0_logits': end_arg0_logits,
            'start_arg1_logits': start_arg1_logits,
            'end_arg1_logits': end_arg1_logits,
            'start_sig_logits': start_sig_logits,
            'end_sig_logits': end_sig_logits,
            'signal_classification_logits': signal_classification_logits,
            'arg0_loss': arg0_loss,
            'arg1_loss': arg1_loss,
            'sig_loss': sig_loss,
            'signal_classification_loss': signal_classification_loss,
            'loss': total_loss,
        }

    def signal_phrases_bias(self, input_ids, signal_bias_mask):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds[signal_bias_mask == 1] += self.signal_phrases_layer  # self.signal_phrases_layer(inputs_embeds[signal_bias_mask == 1])

        return inputs_embeds
    
    def position_selector(
        self,
        start_cause_logits, 
        start_effect_logits, 
        end_cause_logits, 
        end_effect_logits,
        attention_mask,
        word_ids,
    ):
        # basic post processing (removing logits from [CLS], [SEP], [PAD])
        start_cause_logits -= (1 - attention_mask) * 1e4
        end_cause_logits -= (1 - attention_mask) * 1e4
        start_effect_logits -= (1 - attention_mask) * 1e4
        end_effect_logits -= (1 - attention_mask) * 1e4

        start_cause_logits[0] = -1e4
        end_cause_logits[0] = -1e4
        start_effect_logits[0] = -1e4
        end_effect_logits[0] = -1e4

        start_cause_logits[len(word_ids) - 1] = -1e4
        end_cause_logits[len(word_ids) - 1] = -1e4
        start_effect_logits[len(word_ids) - 1] = -1e4
        end_effect_logits[len(word_ids) - 1] = -1e4

        start_cause_logits = torch.log(torch.softmax(start_cause_logits, dim=-1))
        end_cause_logits = torch.log(torch.softmax(end_cause_logits, dim=-1))
        start_effect_logits = torch.log(torch.softmax(start_effect_logits, dim=-1))
        end_effect_logits = torch.log(torch.softmax(end_effect_logits, dim=-1))

        max_arg0_before_arg1 = None
        for i in range(len(end_cause_logits)):
            if attention_mask[i] == 0:
                break
            for j in range(i + 1, len(start_effect_logits)):
                if attention_mask[j] == 0:
                    break

                if max_arg0_before_arg1 is None:
                    max_arg0_before_arg1 = ((i, j), end_cause_logits[i] + start_effect_logits[j])
                else:
                    if end_cause_logits[i] + start_effect_logits[j] > max_arg0_before_arg1[1]:
                        max_arg0_before_arg1 = ((i, j), end_cause_logits[i] + start_effect_logits[j])
        
        max_arg0_after_arg1 = None
        for i in range(len(end_effect_logits)):
            if attention_mask[i] == 0:
                break
            for j in range(i + 1, len(start_cause_logits)):
                if attention_mask[j] == 0:
                    break
                if max_arg0_after_arg1 is None:
                    max_arg0_after_arg1 = ((i, j), start_cause_logits[j] + end_effect_logits[i])
                else:
                    if start_cause_logits[j] + end_effect_logits[i] > max_arg0_after_arg1[1]:
                        max_arg0_after_arg1 = ((i, j), start_cause_logits[j] + end_effect_logits[i])

        if max_arg0_before_arg1[1].item() > max_arg0_after_arg1[1].item():
            end_cause, start_effect = max_arg0_before_arg1[0]
            start_cause_logits[end_cause + 1:] = -1e4
            start_cause = start_cause_logits.argmax().item()

            end_effect_logits[:start_effect] = -1e4
            end_effect = end_effect_logits.argmax().item()
        else:
            end_effect, start_cause = max_arg0_after_arg1[0]
            end_cause_logits[:start_cause] = -1e4
            end_cause = end_cause_logits.argmax().item()

            start_effect_logits[end_effect + 1:] = -1e4
            start_effect = start_effect_logits.argmax().item()
        
        return start_cause, end_cause, start_effect, end_effect


    def beam_search_position_selector(
        self,
        start_cause_logits, 
        start_effect_logits, 
        end_cause_logits, 
        end_effect_logits,
        attention_mask,
        word_ids,
        topk=5
    ):
        # basic post processing (removing logits from [CLS], [SEP], [PAD])

        start_cause_logits -= (1 - attention_mask) * 1e4
        end_cause_logits -= (1 - attention_mask) * 1e4
        start_effect_logits -= (1 - attention_mask) * 1e4
        end_effect_logits -= (1 - attention_mask) * 1e4

        start_cause_logits[0] = -1e4
        end_cause_logits[0] = -1e4
        start_effect_logits[0] = -1e4
        end_effect_logits[0] = -1e4

        start_cause_logits[len(word_ids) - 1] = -1e4
        end_cause_logits[len(word_ids) - 1] = -1e4
        start_effect_logits[len(word_ids) - 1] = -1e4
        end_effect_logits[len(word_ids) - 1] = -1e4

        start_cause_logits = torch.log(torch.softmax(start_cause_logits, dim=-1))
        end_cause_logits = torch.log(torch.softmax(end_cause_logits, dim=-1))
        start_effect_logits = torch.log(torch.softmax(start_effect_logits, dim=-1))
        end_effect_logits = torch.log(torch.softmax(end_effect_logits, dim=-1))

        scores = dict()
        for i in range(len(end_cause_logits)):
            if attention_mask[i] == 0:
                break
            for j in range(i + 1, len(start_effect_logits)):
                if attention_mask[j] == 0:
                    break
                scores[str((i, j, "before"))] = end_cause_logits[i].item() + start_effect_logits[j].item()
        
        for i in range(len(end_effect_logits)):
            if attention_mask[i] == 0:
                break
            for j in range(i + 1, len(start_cause_logits)):
                if attention_mask[j] == 0:
                    break
                scores[str((i, j, "after"))] = start_cause_logits[j].item() + end_effect_logits[i].item()
        
        
        topk_scores = dict()
        for i, (index, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]):
            if eval(index)[2] == 'before':
                end_cause = eval(index)[0]
                start_effect = eval(index)[1]

                this_start_cause_logits = start_cause_logits.clone()
                this_start_cause_logits[end_cause + 1:] = -1e9
                start_cause_values, start_cause_indices = this_start_cause_logits.topk(topk)

                this_end_effect_logits = end_effect_logits.clone()
                this_end_effect_logits[:start_effect] = -1e9
                end_effect_values, end_effect_indices = this_end_effect_logits.topk(topk)

                for m in range(len(start_cause_values)):
                    for n in range(len(end_effect_values)):
                        topk_scores[str((start_cause_indices[m].item(), end_cause, start_effect, end_effect_indices[n].item()))] = score + start_cause_values[m].item() + end_effect_values[n].item()

            elif eval(index)[2] == 'after':
                start_cause = eval(index)[1]
                end_effect = eval(index)[0]

                this_end_cause_logits = end_cause_logits.clone()
                this_end_cause_logits[:start_cause] = -1e9
                end_cause_values, end_cause_indices = this_end_cause_logits.topk(topk)

                this_start_effect_logits = start_effect_logits.clone()
                this_start_effect_logits[end_effect + 1:] = -1e9
                start_effect_values, start_effect_indices = this_start_effect_logits.topk(topk)

                for m in range(len(end_cause_values)):
                    for n in range(len(start_effect_values)):
                        topk_scores[str((start_cause, end_cause_indices[m].item(), start_effect_indices[n].item(), end_effect))] = score + end_cause_values[m].item() + start_effect_values[n].item()

        first, second = sorted(topk_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        return eval(first[0]), eval(second[0]), first[1], second[1], topk_scores
