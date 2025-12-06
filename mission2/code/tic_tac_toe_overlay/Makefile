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
