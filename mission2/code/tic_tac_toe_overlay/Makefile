HF_STAGE ?= /tmp/hf_release
HF_MODEL_REPO ?= your-org/tic-tac-toe-cell-detector
HF_DATASET_REPO ?= your-org/tic-tac-toe-board-dataset

.PHONY: train
train:
	uv run python main.py \
		--image real/images/top-camera05.jpg \
		--output overlay_rocm.jpg \
		--force-regen --force-train \
		--epochs 20 --conf 0.5 \
		--device cuda \
		--grayscale --clahe --contrast-alpha 0.8 --contrast-beta -2 \
		--max-det 30 --train-nms-time 5 \
		--num-train 600 --num-val 120 \
		--real-data real \
		--preprocess-train \
		--save-preprocessed tmp/pre_rocm.jpg

.PHONY: hf-stage-clean
hf-stage-clean:
	rm -rf $(HF_STAGE)

# Stage only model artifacts (weights + README/license). Codebase stays private.
.PHONY: hf-model-stage
hf-model-stage: hf-stage-clean
	rm -rf $(HF_STAGE)/model
	git clone https://huggingface.co/$(HF_MODEL_REPO) $(HF_STAGE)/model
	cd $(HF_STAGE)/model && git checkout main
	cp MODEL_CARD.md $(HF_STAGE)/model/README.md
	cp models/train/weights/best.pt $(HF_STAGE)/model/
	printf "CC-BY-4.0\n" > $(HF_STAGE)/model/LICENSE

# Stage only dataset artifacts (real images/labels + README/license).
.PHONY: hf-dataset-stage
hf-dataset-stage: hf-stage-clean
	rm -rf $(HF_STAGE)/dataset
	git clone https://huggingface.co/datasets/$(HF_DATASET_REPO) $(HF_STAGE)/dataset
	cd $(HF_STAGE)/dataset && git checkout main
	cp DATASET_CARD.md $(HF_STAGE)/dataset/README.md
	rsync -a real/ $(HF_STAGE)/dataset/real/
	uv run python export_parquet.py --images $(HF_STAGE)/dataset/real/images --labels $(HF_STAGE)/dataset/real/labels --out $(HF_STAGE)/dataset/dataset.parquet
	printf "CC-BY-4.0\n" > $(HF_STAGE)/dataset/LICENSE

# Push to Hugging Face (requires huggingface-cli login). Repo creation is idempotent.
.PHONY: hf-model-push
hf-model-push: hf-model-stage
	cd $(HF_STAGE)/model && \
	git lfs install --local || true && \
	git add . && git commit -m "Update model weights (CC-BY-4.0)" || true && \
	HF_HUB_ENABLE_HF_TRANSFER=1 git push origin main

.PHONY: hf-dataset-push
hf-dataset-push: hf-dataset-stage
	cd $(HF_STAGE)/dataset && \
	git lfs install --local || true && \
	git add . && git commit -m "Update dataset (CC-BY-4.0)" || true && \
	HF_HUB_ENABLE_HF_TRANSFER=1 git push origin main

hf-dataset-inspect:
	@uv run python -c "import pyarrow.parquet as pq; from pathlib import Path; path=Path('$(HF_STAGE)/dataset/dataset.parquet'); \
print(f'[info] path: {path}'); \
if not path.exists(): \
    print('[error] file not found; run `make hf-dataset-stage` first'); \
    raise SystemExit(1); \
table = pq.read_table(path); \
print('[info] schema:', table.schema); \
print('[info] rows:', table.num_rows); \
head = table.slice(0, min(5, table.num_rows)).to_pydict(); \
print('[info] head:', head); \
if 'class_id' in table.column_names: \
    uniq = table.column('class_id').unique().to_pylist(); \
    print('[info] distinct class_id:', uniq)" || true
