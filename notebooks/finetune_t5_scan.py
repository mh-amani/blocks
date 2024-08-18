import code
t = 0.04
epochs = 10
bsize=32
max_length=128

from datasets import load_dataset
# Load the SCAN dataset
dataset = load_dataset('scan', 'simple')

from sklearn.model_selection import train_test_split
# Split the original training data into training and validation sets
train_val_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
# Subsample 10% of the training data
train_dataset = train_dataset.train_test_split(test_size=t)['test']

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-large')
def preprocess_function(examples):
    inputs = [ex for ex in examples['commands']]
    targets = [ex for ex in examples['actions']]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)


import torch
# import evaluate
# accuracy_metric = evaluate.load("accuracy")
# sacrebleu_metric = evaluate.load("sacrebleu")

import numpy as np
def compute_metrics(pred):
    # print(pred)
    # code.interact(local=locals())
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Pad predictions to match the length of labels
    max_label_len = labels_ids.shape[1]
    pred_ids_padded = np.zeros((pred_ids.shape[0], max_label_len), dtype=np.int32)
    pred_ids_padded[:, :pred_ids.shape[1]] = pred_ids


    # print(labels_ids)
    # print(pred_ids)
    # Decode generated texts
    # pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    labels_ids = labels_ids.astype(np.int32)[:, :-1]
    pred_ids_padded = pred_ids_padded.astype(np.int32)[:,1:]
    pred_ids_padded[labels_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id
    # label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Calculate BLEU Score
    # sacrebleu_results = sacrebleu_metric.compute(predictions=pred_str, references=[[ref] for ref in label_str])
    # code.interact(local=locals())
    # Calculate Accuracy: Token-level accuracy
    equality_matrix = labels_ids == pred_ids_padded
    num_pad_labels = np.sum(labels_ids == tokenizer.pad_token_id)

    accuracy = (np.sum(equality_matrix) - num_pad_labels) / (np.prod(labels_ids.shape) - num_pad_labels)

    # Calculate Sentence Accuracy
    sentence_accuracy = np.sum(np.prod(equality_matrix, axis=1)) / pred_ids_padded.shape[0]

    return {
        "accuracy": accuracy,
        # "sacrebleu": sacrebleu_results["score"],
        "sentence_accuracy": sentence_accuracy,
    }


# to debug
# import torch
# input = torch.tensor(tokenized_train_dataset[0]['input_ids'], device=model.device).unsqueeze(0)
# label = torch.tensor(tokenized_train_dataset[0]['labels'], device=model.device).unsqueeze(0)
# out = model(input, labels=label)
# out.loss
# tokenizer.decode(out.logits.argmax(-1)[0])

from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=bsize,
    per_device_eval_batch_size=bsize,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True,
    generation_max_length=128,  # Set the max length for generation
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

print(results)
