import pandas as pd
from transformers import TrainingArguments

from src.utils.dataset_aligner import DatasetAligner
from fine_tuning_system.training.label_schema import LabelSchema
from fine_tuning_system.fine_tune.roberta_experiment import RobertaExperiment

from src.utils.data_loader import load_hf_dataset_as_dataframe

dataset_name = "rotten_tomatoes"  # exemplo
cache_dir = "./cache"

df_gt, categories = load_hf_dataset_as_dataframe(
    dataset_name=dataset_name,
    cache_dir=cache_dir,
    dataset_global_config=DATASET_GLOBAL_CONFIG,
)

label_schema = LabelSchema.from_dataframe(df_gt)

df_ann = pd.read_csv("dataset_anotado_completo.csv")

aligner = DatasetAligner(
    text_column="text",
    gt_label="label",
    ann_label="most_common_annotation"
)

df_gt_aligned, df_ann_aligned = aligner.align(df_gt, df_ann)

assert (df_gt_aligned["text"] == df_ann_aligned["text"]).all()

from datasets import Dataset

train_gt = Dataset.from_pandas(df_gt_aligned)
train_consensus = Dataset.from_pandas(df_ann_aligned)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

experiment = RobertaExperiment(
    model_name="roberta-base",
    training_args=training_args,
    label_schema=label_schema
)

results = []

results.append(
    experiment.run(train_consensus, train_gt, "consensus_llm")
)

results.append(
    experiment.run(train_gt, train_gt, "ground_truth")
)

print(pd.DataFrame(results))


