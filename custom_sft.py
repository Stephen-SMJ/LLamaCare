from ctypes import Union
from typing import Optional, Dict

from datasets import Dataset
from torch import nn
from trl import SFTTrainer, is_peft_available
from functools import wraps

from trl.trainer.utils import PeftSavingCallback, DataCollatorForCompletionOnlyLM, ConstantLengthDataset, \
    neftune_post_forward_hook

from train_fn import CustomTrainer
import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


class NewSftTrainer(CustomTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module, str] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional["PeftConfig"] = None,
            dataset_text_field: Optional[str] = None,
            packing: Optional[bool] = False,
            formatting_func: Optional[Callable] = None,
            max_seq_length: Optional[int] = None,
            infinite: Optional[bool] = False,
            num_of_sequences: Optional[int] = 1024,
            chars_per_token: Optional[float] = 3.6,
            dataset_num_proc: Optional[int] = None,
            dataset_batch_size: int = 1000,
            neftune_noise_alpha: Optional[float] = None,
            model_init_kwargs: Optional[Dict] = None,
            clf_weight: Optional[Dict] = 0.2,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SFTTrainer. But your model is already instantiated.")

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                    _support_gc_kwargs = hasattr(
                        args, "gradient_checkpointing_kwargs"
                    ) and "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if _support_gc_kwargs:
                        preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)

                    args = dataclasses.replace(args, gradient_checkpointing=False)

                model = get_peft_model(model, peft_config)
                for name, param in model.base_model.model.classification_head.named_parameters():
                    # print(name, param.numel(), param.requires_grad)
                    if 'fc' in name:
                        param.requires_grad = True
                        # print("{}:{}".format(name, param))
                # model.base_model.model.classification_head.weight.requires_grad = True
                # for name, param in model.named_parameters():
                #     # print(name, param.numel(), param.requires_grad)
                #     if name == "base_model.model.classification_head.linear.weight":
                #         print("{}:{}".format(name,param))
                #         param.requires_grad = True
                # print()
            if callbacks is None:
                callbacks = [PeftSavingCallback]

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        if max_seq_length is None:
            # to overcome some issues with broken tokenizers
            max_seq_length = min(tokenizer.model_max_length, 1024)

            warnings.warn(
                f"You didn't pass a `max_seq_length` argument to the SFTTrainer, this will default to {max_seq_length}"
            )

        self.dataset_num_proc = dataset_num_proc
        self.dataset_batch_size = dataset_batch_size
        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")

        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override the one in the `TrainingArguments`."
            )
            # self.neftune_noise_alpha is done at Trainer level
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if not packing:
            if dataset_text_field is None and formatting_func is None:
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer, but you didn't pass a `dataset_text_field` or `formatting_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                train_dataset,
                tokenizer,
                packing,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                infinite,
                num_of_sequences,
                chars_per_token,
            )
        if eval_dataset is not None:
            eval_dataset = self._prepare_dataset(
                eval_dataset,
                tokenizer,
                packing,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                infinite,
                num_of_sequences,
                chars_per_token,
            )

        if tokenizer.padding_side is not None and tokenizer.padding_side != "right":
            warnings.warn(
                "You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to "
                "overflow issues when training a model in half-precision. You might consider adding `tokenizer.padding_side = 'right'` to your code."
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            # max_seq_length=max_seq_length,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            #clf_weight=clf_weight
        )

        if self.args.max_steps > 0 and packing:
            warnings.warn(
                "You passed `packing=True` to the SFTTrainer, and you are training your model with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and packing:
            self.train_dataset.infinite = False

    @wraps(CustomTrainer.train)
    def train(self, *args, **kwargs):
        # Activate neftune right before training.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            self.model = self._trl_activate_neftune(self.model)
        # print(self.custom_trainer)
        output = super().train(*args, **kwargs)
        # output = self.custom_trainer.train(*args, **kwargs) #在这里调用train

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            unwrapped_model = unwrap_model(self.model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                embeddings = unwrapped_model.base_model.model.get_input_embeddings()
            else:
                embeddings = unwrapped_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            del embeddings.neftune_noise_alpha

        return output

    def _prepare_dataset(
            self,
            dataset,
            tokenizer,
            packing,
            dataset_text_field,
            max_seq_length,
            formatting_func,
            infinite,
            num_of_sequences,
            chars_per_token,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        # check if torch dataset / dataloader and do nothing
        if isinstance(dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)):
            return dataset

        if not packing:
            return self._prepare_non_packed_dataloader(
                tokenizer, dataset, dataset_text_field, max_seq_length, formatting_func
            )

        if dataset_text_field is not None or formatting_func is not None:
            if tokenizer is None:
                raise ValueError(
                    "You need to pass a tokenizer when using the SFT Trainer when passing a `dataset_text_field`."
                )

            return ConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=infinite,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
            )

        raise ValueError(
            "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
        )

    def _prepare_non_packed_dataloader(
            self, tokenizer, dataset, dataset_text_field, max_seq_len, formatting_func=None
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                truncation=True,
                padding=False,
                max_length=max_seq_len,
                return_overflowing_tokens=False,
                return_length=False,
            )

            #label = [1 if element['final_decision'] == 'yes' else 0]
            label = []
            for i in element['final_decision']:
                if i == "no":
                    label.append(0)
                elif i == "yes":
                    label.append(1)
                else:
                    label.append(2)

            ctl = torch.tensor(label, dtype=torch.long)

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "ctl": ctl}

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

    def _trl_activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        """
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model
