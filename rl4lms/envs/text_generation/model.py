from transformers import AutoModelForCausalLM, CausalLMOutput
import torch.nn as nn

class CustomCausalLMWithClassificationHead(AutoModelForCausalLM):
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.classification_head = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, labels=None, decision_labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, labels=labels, **kwargs)
        sequence_output = outputs.last_hidden_state
        logits = self.classification_head(sequence_output[:, 0])

        # Handle the decision labels and calculate the decision loss if provided
        decision_loss = None
        if decision_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            decision_loss = loss_fct(logits, decision_labels)

        # Return the outputs in a structured way
        return CausalLMOutput(
            loss=outputs.loss + decision_loss if outputs.loss is not None and decision_loss is not None else None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            decision_logits=logits,
            decision_loss=decision_loss
        )

from transformers import AutoModelForSeq2SeqLM, Seq2SeqLMOutput
import torch.nn as nn

class CustomSeq2SeqLMWithClassificationHead(AutoModelForSeq2SeqLM):
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.classification_head = nn.Linear(config.d_model, num_labels)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decision_labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        sequence_output = outputs.encoder_last_hidden_state
        logits = self.classification_head(sequence_output[:, 0])

        # Handle the decision labels and calculate the decision loss if provided
        decision_loss = None
        if decision_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            decision_loss = loss_fct(logits, decision_labels)

        # Return the outputs in a structured way
        return Seq2SeqLMOutput(
            loss=outputs.loss + decision_loss if outputs.loss is not None and decision_loss is not None else None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            decision_logits=logits,
            decision_loss=decision_loss
        )
