PYTHON ?= python
DATA_DIR ?= data
RAW_DIR := $(DATA_DIR)/raw
PAIRS := $(DATA_DIR)/pairs.jsonl
AUG := $(DATA_DIR)/aug.jsonl
HF_DIR := $(DATA_DIR)/hf
CONFIG ?= configs/base.yaml
CHECKPOINT ?= checkpoints/best
SAMPLES ?= $(CHECKPOINT)/samples.txt

.PHONY: setup fetch extract augment build train eval clean

setup:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

fetch:
	$(PYTHON) scripts/fetch_data.py --dest $(RAW_DIR)

extract:
	$(PYTHON) scripts/extract_pairs.py --src $(RAW_DIR) --out $(PAIRS)

augment:
	$(PYTHON) scripts/augment_inputs.py --in $(PAIRS) --out $(AUG) --nlx 8

build:
	$(PYTHON) scripts/build_hf_dataset.py --in $(AUG) --out $(HF_DIR)

train:
	$(PYTHON) scripts/train_lora.py --config $(CONFIG)

eval:
	$(PYTHON) scripts/evaluate.py --model $(CHECKPOINT) --data $(HF_DIR) --split val --samples $(SAMPLES)

clean:
	rm -rf $(DATA_DIR) checkpoints logs
