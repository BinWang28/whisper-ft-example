
import torch
import time

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch.distributed as dist


from datasets import load_dataset, DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

import evaluate
import os

os.environ["WANDB_PROJECT"] = "WHISPER"
os.environ["WANDB_ENTITY"]  = "i2r-llm"

# Dataset
# common_voice = DatasetDict()
# common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
# common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True)
# common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
# common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

imda_nsc_data = DatasetDict()

# Training Data
part_1_data = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/train/ASR/IMDA_PART1_ASR_v3")
part_1_data = part_1_data.remove_columns(["instruction", "other_attributes"])

part_2_data = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/train/ASR/IMDA_PART2_ASR_v3")
part_2_data = part_2_data.remove_columns(["instruction", "other_attributes"])

part_3_data = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/train/ASR/IMDA_PART3_30_ASR_v2")
part_3_data = part_3_data.remove_columns(["instruction", "other_attributes"])

part_4_data = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/train/ASR/IMDA_PART4_30_ASR_v2")
part_4_data = part_4_data.remove_columns(["instruction", "other_attributes"])

part_5_data = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/train/ASR/IMDA_PART5_30_ASR_v2")
part_5_data = part_5_data.remove_columns(["instruction", "other_attributes"])

part_6_data = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/train/ASR/IMDA_PART6_30_ASR_v2")
part_6_data = part_6_data.remove_columns(["instruction", "other_attributes"])

# Test Data
part_1_data_test = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/test/ASR/IMDA_PART1_ASR_v2")
part_1_data_test = part_1_data_test.remove_columns(["instruction", "other_attributes"])

part_2_data_test = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/test/ASR/IMDA_PART2_ASR_v2")
part_2_data_test = part_2_data_test.remove_columns(["instruction", "other_attributes"])

part_3_data_test = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/test/ASR/IMDA_PART3_30_ASR_v2")
part_3_data_test = part_3_data_test.remove_columns(["instruction", "other_attributes"])

part_4_data_test = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/test/ASR/IMDA_PART4_30_ASR_v2")
part_4_data_test = part_4_data_test.remove_columns(["instruction", "other_attributes"])

part_5_data_test = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/test/ASR/IMDA_PART5_30_ASR_v2")
part_5_data_test = part_5_data_test.remove_columns(["instruction", "other_attributes"])

part_6_data_test = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/test/ASR/IMDA_PART6_30_ASR_v2")
part_6_data_test = part_6_data_test.remove_columns(["instruction", "other_attributes"])

# imda_nsc_data["train"] = concatenate_datasets([part_1_data, part_2_data, part_3_data, part_4_data, part_5_data, part_6_data]).shuffle(seed=42)
# imda_nsc_data["test"] = concatenate_datasets([part_1_data_test, part_2_data_test, part_3_data_test, part_4_data_test, part_5_data_test, part_6_data_test])

imda_nsc_data["train"] = concatenate_datasets([part_4_data]).shuffle(seed=42)
imda_nsc_data["test"] = concatenate_datasets([part_4_data_test])

run_name = 'whisper-ft-part1_2'
run_name = 'whisper-ft-part1_2_3_run2'
run_name = 'whisper-ft-part1_2_3_5_6'
run_name = 'whisper-ft-part4'







# Test data

#imda_nsc_data["train"]  = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/test/ASR/IMDA_PART2_ASR_v2")
#imda_nsc_data["test"]   = load_from_disk("/home/users/astar/ares/wangb1/scratch/workspaces_wb/AudioLLM/data/imda_experiments_hf_opus/test/ASR/IMDA_PART2_ASR_v2")
#imda_nsc_data           = imda_nsc_data.remove_columns(["instruction", "other_attributes"])
                                              
# breakpoint()


#imda_nsc_data = imda_nsc_data.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
#imda_nsc_data = imda_nsc_data.cast_column("audio", Audio(sampling_rate=16000))

#breakpoint()

# Model
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
tokenizer         = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="English", task="transcribe")
processor         = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="English", task="transcribe")


do_lower_case         = False
do_remove_punctuation = False
normalizer            = BasicTextNormalizer()

def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    #audio         = batch["audio"]
    #transcription = batch["sentence"]

    audio         = batch["context"]["audio"]
    transcription = batch["answer"]["text"]


    if len(audio['array']) / 16000 >= 30:
        print("Audio length greater than 30 seconds")
        return None

    if len(audio['array']) / 16000 == 0:
        print("Audio length is 0")
        return None

    

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    if do_lower_case:
        transcription = transcription.lower()
    
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids

    if len(batch["labels"]) > 300:
        print("Transcription length greater than 300")
        return None

    return batch

# common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=32)
imda_nsc_data = imda_nsc_data.map(prepare_dataset, remove_columns=imda_nsc_data.column_names["train"], num_proc=32)


max_input_length = 30.0
def is_audio_in_length_range(length):
    return length < max_input_length

#common_voice["train"] = common_voice["train"].filter(
#    is_audio_in_length_range,
#    input_columns=["input_length"],
#)

imda_nsc_data["train"] = imda_nsc_data["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)



#breakpoint()

#common_voice.save_to_disk("data/common_voice_11_0_hi")
#common_voice = load_dataset("data/common_voice_11_0_hi", streaming=True)

#breakpoint()




@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric        = evaluate.load("wer")

# evaluate with the 'normalised' WER
do_normalize_eval = True

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # print("00000000000")
    # print(time.time())


    # we do not want to group tokens when computing the metrics
    pred_str  = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # print("11111111111")
    # print(time.time())

    if do_normalize_eval:
        pred_str  = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    # print("22222222222")
    # print(time.time())

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    # print("33333333333")
    # print(time.time())

    # Only save predictions on rank 1 (or the main process)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(f"predictions/{run_name}", exist_ok=True)
        with open(f"predictions/{run_name}/predictions_and_labels_{wer}.txt", "w") as f:
            for pred, label in zip(pred_str, label_str):
                f.write(f"Prediction: {pred}\n")
                f.write(f"Reference: {label}\n")
                f.write("\n")  # Add spacing between entries

    return {"wer": wer}


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

model.config.forced_decoder_ids  = None
model.config.suppress_tokens     = []
model.config.use_cache           = False
#model.generation_config.language = 'en'

model.generation_config.language = "<|en|>"
model.generation_config.task = "transcribe"


training_args = Seq2SeqTrainingArguments(
    output_dir                  = f"./trained_model/{run_name}",
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    gradient_accumulation_steps = 1,                            # increase by 2x for every 2x decrease in batch size
    learning_rate               = 1e-5,
    warmup_steps                = 50,
    num_train_epochs            = 5,
    gradient_checkpointing      = True,
    fp16                        = True,
    eval_strategy               = "steps",
    predict_with_generate       = True,
    generation_max_length       = 225,
    save_steps                  = 500,
    eval_steps                  = 500,
    logging_steps               = 10,
    report_to                   = ["wandb"],
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    push_to_hub                 = False,
    run_name                    = run_name,
    dataloader_num_workers      = 32,
    save_safetensors            = True,
    dataloader_prefetch_factor  = 8,
    eval_on_start               = True,
)


trainer = Seq2SeqTrainer(
    args            = training_args,
    model           = model,
    #train_dataset   = common_voice["train"],
    #eval_dataset    = common_voice["test"],
    train_dataset   = imda_nsc_data["train"],
    eval_dataset    = imda_nsc_data["test"],
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    tokenizer       = processor.feature_extractor,
)

feature_extractor.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

trainer.train()

# Save the model
trainer.save_model(f'./trained_model/{run_name}/best_model_at_end')


# kwargs = {
#     "dataset_tags": "mozilla-foundation/common_voice_11_0",
#     "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
#     "language": "hi",
#     "model_name": "Whisper Small Hi - Sanchit Gandhi",  # a 'pretty' name for your model
#     "finetuned_from": "openai/whisper-small",
#     "tasks": "automatic-speech-recognition",
#     "tags": "whisper-event",
# }

# trainer.push_to_hub(**kwargs)
