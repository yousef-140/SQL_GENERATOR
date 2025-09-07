from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

ds = load_dataset("xlangai/spider")

def format_example(example):
    return {"text": f"Question: {example['question']}\nSQL: {example['query']}"}

train_data = ds["train"].map(format_example)
val_data = ds["validation"].map(format_example)

print(train_data[0])


model_id = "microsoft/phi-2"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

train_tokenized = train_data.map(tokenize_fn, batched=True, remove_columns=train_data.column_names)
val_tokenized = val_data.map(tokenize_fn, batched=True, remove_columns=val_data.column_names)


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./phi2-sql-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    num_train_epochs=3,
    learning_rate=1e-5,
    fp16=True,
    report_to="none",
    save_total_limit=2,
    metric_for_best_model="eval_loss"

)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()


