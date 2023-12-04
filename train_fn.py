import gc
import torch
from torch import nn
from transformers import Trainer, LlamaModel
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
# from network import ClassificationHead
import torch.nn.functional as F
from param_dict import clf_weight, logging_steps, output_dir
import json
from plt import save_state, plot_clf_loss

class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.clf_log_history = []
        # clf_wight = kwargs.pop("clf_weight")
        # input_dim = kwargs["max_seq_length"]
        # kwargs.pop("max_seq_length")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False): #override

        classification_labels = inputs.pop("ctl")

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] #original_loss

        # compute a classification loss here.
        last_hidden_state = outputs["hidden_states"][-1] #get output embedding of last decoder #[2,32,4096]
        embedding_flatten = last_hidden_state.view(last_hidden_state.shape[0], last_hidden_state.shape[1]*last_hidden_state.shape[2]) #[2,131072]
        classification_output = model.classification_head(embedding_flatten) # [2,3]
        loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss has a softmax function inside, so we don't need softmax the output.
        classification_loss = loss_fn(classification_output, classification_labels)
        #print("classification weight:", model.classification_head.fc1.weight)
        # print()
        if self.state.global_step % logging_steps == 0:
            clf_state = {
                "epoch": self.state.epoch,
                "step": self.state.global_step,
                "loss": classification_loss.item()
            }
            self.clf_log_history.append(clf_state)
        if self.state.global_step == self.state.max_steps-1:
            print(self.clf_log_history)
            save_state(self.clf_log_history)
            plot_clf_loss(output_dir,["loss"])

        #combine_loss
        combined_loss = (1-clf_weight) * loss + (clf_weight * classification_loss)
        # print("classification loss:{} | combined loss:{} ".format(classification_loss,combined_loss))

        return (combined_loss, outputs) if return_outputs else combined_loss

