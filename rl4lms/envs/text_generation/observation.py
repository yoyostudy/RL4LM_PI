from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer
from rl4lms.data_pools.text_generation_pool import Sample
from copy import deepcopy

@dataclass
class Observation:
    # encoded input
    prompt_or_input_encoded_pt: torch.tensor
    # attention mask for the input
    prompt_or_input_attention_mask_pt: torch.tensor
    # input text
    prompt_or_input_text: str
    # encoded context
    context_encoded_pt: torch.tensor
    # attention mask for the context
    context_attention_mask_pt: torch.tensor
    # context text
    context_text: str
    # reference texts
    target_or_reference_texts: List[str]

    # concatenated input
    input_encoded_pt: torch.tensor
    input_attention_mask_pt: torch.tensor

    # list of actions
    action_history: List[str]

    # TODO: add policy_chosen
    policy_chosen: int # -1 for not chosen yet

    # other meta info
    meta_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, torch.tensor]:
        """
        For stable baselines (only return tensor items)
        """
        dict_obs = {
            "prompt_or_input_encoded_pt": self.prompt_or_input_encoded_pt.numpy().flatten(),
            "prompt_or_input_attention_mask_pt": self.prompt_or_input_attention_mask_pt.numpy().flatten(),
            "context_encoded_pt": self.context_encoded_pt.numpy().flatten(),
            "context_attention_mask_pt": self.context_attention_mask_pt.numpy().flatten(),
            "input_encoded_pt": self.input_encoded_pt.numpy().flatten(),
            "input_attention_mask_pt": self.input_attention_mask_pt.numpy().flatten(),
            # TODO: should I add policy_chosen here?
        }
        return dict_obs

    @staticmethod
    def _concat(prompt: torch.tensor, prompt_mask: torch.tensor,
                context: torch.tensor, context_mask: torch.tensor,
                pad_token: int, policy_chosen: int = -1):
        
        # TODO: add policy_chosen
        policy_chosen = torch.tensor([policy_chosen]).expand(1, -1)

        prompt_ = prompt[:, prompt_mask.flatten().bool().tolist()]
        context_ = context[:, context_mask.flatten().bool().tolist()]

        # TODO: add policy_chosen to actual size
        #actual_size = prompt_.shape[1] + context_.shape[1]
        actual_size = prompt_.shape[1] + context_.shape[1] + policy_chosen.shape[1]

        # TODO: add policy_chosen to full size
        #full_size = prompt.shape[1] + context.shape[1]
        full_size = prompt.shape[1] + context.shape[1] + policy_chosen.shape[1]

        concatenated = torch.full(
            (full_size,), fill_value=pad_token).reshape(1, -1)
        concatenated_mask = torch.zeros((1, full_size)).int()

        # TODO: add policy_chosen
        # concatenated[:, full_size -
        #              actual_size:] = torch.cat((prompt_, context_), dim=1)
        concatenated[:, full_size -
                        actual_size:] = torch.cat((prompt_, context_, policy_chosen), dim=1)
        concatenated_mask[:, full_size -
                          actual_size:] = 1
        return concatenated, concatenated_mask


    ## TODO: add two head action
    def update(self, policy_chosen: int, action: int, tokenizer: AutoTokenizer) -> "Observation":
        """
        Updates the observation using the given action

        # TODO: add two head action
        policy_chosen: int, -1 for not chosen yet, 0 for Seq2Seq, 1 for Casual
        action: int, the token id of the action
        """

        # update the action history
        # TODO: add two head action, add policy_chosen head
        current_action_history = deepcopy(self.action_history)
        policy_token = tokenizer._convert_id_to_token(policy_chosen)
        action_token = tokenizer._convert_id_to_token(action)
        current_action_history.append((policy_token, action_token))

        # update the policy_chosen
        # TODO: make sure the policy only update if it is not chosen yet
        assert (self.policy_chosen == -1 or self.policy_chosen == policy_chosen), "policy update not correct!"
        self.policy_chosen = policy_chosen

        # get the current context
        current_context = deepcopy(self.context_encoded_pt)
        current_context_attention_mask = deepcopy(
            self.context_attention_mask_pt)

        # just shift the context (also the attention mask) to left by 1
        current_context[:, 0:-1] = current_context[:, 1:].clone()
        current_context_attention_mask[:, 0:-
                                       1] = current_context_attention_mask[:, 1:].clone()

        # add the action always at the end (assumes left padding)
        current_context[:, -1] = action
        current_context_attention_mask[:, -1] = 1

        # decode the context
        context_text = tokenizer.decode(
            current_context.flatten(), skip_special_tokens=True)

        # concatenate and still keep the left padding
        # TODO: add policy_chosen to the obs input
        input_encoded_pt, input_attention_mask_pt = Observation._concat(
            self.prompt_or_input_encoded_pt, self.prompt_or_input_attention_mask_pt,
            current_context, current_context_attention_mask,
            tokenizer.pad_token_id, self.policy_chosen)

        # and create a new observation
        # TODO: add policy_chosen
        obs = Observation(self.prompt_or_input_encoded_pt,
                          self.prompt_or_input_attention_mask_pt,
                          self.prompt_or_input_text,
                          current_context,
                          current_context_attention_mask,
                          context_text,
                          self.target_or_reference_texts,
                          input_encoded_pt,
                          input_attention_mask_pt,
                          current_action_history,
                          self.policy_chosen,
                          self.meta_info)

        return obs

    @ classmethod
    def init_from_sample(cls, sample: Sample,
                         tokenizer: AutoTokenizer,
                         max_input_length: int,
                         max_context_length: int,
                         prompt_truncation_side: str,
                         context_start_token: int = None,
                         policy_chosen: int = -1, # -1 for not chosen yet
                         meta_info: Dict[str, Any] = None):
        # encode the prompt text
        # override truncation side for prompt
        prev_truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = prompt_truncation_side
        prompt_outputs = tokenizer(sample.prompt_or_input_text,
                                   padding="max_length",
                                   max_length=max_input_length,
                                   return_tensors="pt",
                                   return_attention_mask=True,
                                   truncation=True)
        tokenizer.truncation_side = prev_truncation_side
        

        # for seq2seq models, context should be initialized to start token if provided
        if context_start_token is not None:
            context_outputs = tokenizer("",
                                    padding="max_length",
                                    max_length=max_context_length,
                                    return_tensors="pt",
                                    return_attention_mask=True)
            context_outputs.input_ids = torch.ones(1, max_context_length, dtype=torch.int32) * tokenizer.pad_token_id
            context_outputs.input_ids[:, -1] = context_start_token
            context_outputs.attention_mask = torch.zeros(1, max_context_length, dtype=torch.int32)
            context_outputs.attention_mask[:,-1] = 1
        else:
            context_outputs = tokenizer("",
                                    padding="max_length",
                                    max_length=max_context_length,
                                    return_tensors="pt",
                                    return_attention_mask=True)

        # concatenate
        input_encoded_pt, input_attention_mask_pt = Observation._concat(
            prompt_outputs.input_ids, prompt_outputs.attention_mask,
            context_outputs.input_ids, context_outputs.attention_mask,
            tokenizer.pad_token_id)

        obs = Observation(prompt_or_input_encoded_pt=prompt_outputs.input_ids,
                          prompt_or_input_attention_mask_pt=prompt_outputs.attention_mask,
                          prompt_or_input_text=sample.prompt_or_input_text,
                          context_encoded_pt=context_outputs.input_ids,
                          context_attention_mask_pt=context_outputs.attention_mask,
                          input_encoded_pt=input_encoded_pt,
                          input_attention_mask_pt=input_attention_mask_pt,
                          context_text="",
                          target_or_reference_texts=sample.references,
                          action_history=[],
                          policy_chosen=-1, # TODO: add policy_chosen for two-head action, initial set to -1 for not chosen yet
                          meta_info=meta_info)

        return obs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    sample = Sample(
        id = "1", 
        prompt_or_input_text= "Hello, this is cool", 
        references= ["it is good", "going well"],
        meta_data = {"key": "value"})

    print(sample)

    obs = Observation.init_from_sample(
        sample=sample,
        tokenizer=tokenizer,
        max_input_length=24,
        max_context_length=24,
        prompt_truncation_side="left", # add this line 
    )
    # TODO: add two head action for update
    policy_chosen = 0 # action head 1, decision making head
    action = 10 # action head 2, generation head
    updated_obs = obs.update(
        policy_chosen=0, action = 10, tokenizer = tokenizer)
    
    print(updated_obs)
    #updated_obs = updated_obs.update(11, tokenizer)
