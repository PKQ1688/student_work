#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/10 22:44
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/5/10 22:44
# @File         : finetune.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import librosa
import torch
from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor,Wav2Vec2ForCTC,TrainingArguments,Trainer

# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

# Load the dataset from the CSV file
dataset = load_dataset("csv", data_files="filename_to_phonetic.csv")["train"]
print(dataset[0])

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)

# def prepare_example(example):
#     audio_path = f"phonetic_wav/{example['filename']}"
#     speech_array, sr = librosa.load(
#         audio_path, sr=16000
#     )  # Assuming librosa is already importing
#     example["speech"] = {
#         "array": np.array(speech_array, dtype=np.float32),
#         "sampling_rate": sr,
#     }
#     return example


# processor = Wav2Vec2Processor(feature_extractor, tokenizer=None)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


# model = AutoModelForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')


def prepare_dataset(example):
    audio_path = f"phonetic_wav/{example['filename']}"
    # audio = batch["audio"]
    speech_array, sr = librosa.load(
        audio_path, sr=16000
    )  # Assuming librosa is already importing
    # batched output is "un-batched" to ensure mapping is correct
    example["input_values"] = processor(speech_array, sampling_rate=sr).input_values[0]

    with processor.as_target_processor():
        # example["labels"] = processor(example["label"]).input_ids
        example["labels"] = example["label"]
    return example


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
# wer_metric = load_metric("wer")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-960h-lv60-self",
    ctc_loss_reduction="mean",
    # pad_token_id=processor.tokenizer.pad_token_id,
)
model.freeze_feature_extractor()

training_args = TrainingArguments(
  output_dir="result",
  group_by_length=True,
  per_device_train_batch_size=32,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=processor.feature_extractor,
)
#
trainer.train()
#
# model.save_pretrained('./wav2vec2-finetuned')
# processor.save_pretrained('./wav2vec2-finetuned')
