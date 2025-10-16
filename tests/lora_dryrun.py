
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# tiny dataset
ds = Dataset.from_dict({"text":["### CONTEXT\n\n### GOAL\nlemma : true\n### SOLUTION\nby trivial\n"]})
def tokfn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=128)
ds = ds.map(tokfn, batched=False)

# apply LoRA (works as API smoke test)
lora_config = LoraConfig(r=4, lora_alpha=16, target_modules=["q_proj","v_proj"], bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.train()

training_args = TrainingArguments(output_dir="out", per_device_train_batch_size=1, num_train_epochs=1, fp16=False)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
trainer.train()
print("LoRA dry-run complete")
