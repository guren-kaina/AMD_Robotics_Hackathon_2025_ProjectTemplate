# AMD_Robotics_Hackathon_2025_Kaina-TTT

## Team Information
**Team:** Kaina

**Members** [Ryo Igarashi](https://github.com/igaryo0506), [Kenta Mori](https://github.com/zoncoen), [Shota Iwami](https://github.com/BIwashi), [Gen Shu](https://github.com/genkey6)

**Summary**
Tic-tac-toe solver arm that reads the board from a top camera, plans the next move for player **X**, and overlays the target cell for the SO-Arm 101 to execute. We fine-tuned a lightweight YOLOv8n with mostly synthetic grids (plus optional real labels) in `mission2/code/tic_tac_toe_overlay` to classify each cell as empty/O/X. The text-only planner in `mission2/code/tic-tac-toe-planner` uses a local LLM (default `Qwen/Qwen3-4B-Instruct-2507`) to choose the move, avoiding heavy VLA reasoning. `mission2/code/ttt-pipeline` stitches the pieces: capture frames, detect the 3x3 board, call the planner when the board changes, paint the chosen cell, and stream the masked video to a virtual camera. Hardware follows the shared slide deck: top/side/gripper cameras, top light, AMD laptop, SO-Arm 101, and a 3D-printed board fixture.

**How To**: reproduce end-to-end
1. Set up dependencies (Python 3.10+). Using `uv`:
   ```bash
   cd mission2/code/tic_tac_toe_overlay   && uv sync
   cd ../tic-tac-toe-planner              && uv sync
   cd ../ttt-pipeline                     && uv sync
   ```
   Ensure `ffmpeg` and a v4l2loopback/virtual camera device (e.g., `/dev/video12`) exist for streaming. Download the planner model locally (default `Qwen/Qwen3-4B-Instruct-2507`) so transformers can load it offline.
2. Train or reuse the YOLO cell detector. The overlay script auto-generates a synthetic dataset and trains if weights are missing:
   ```bash
   cd mission2/code/tic_tac_toe_overlay
   uv run python3 main.py \
     --image top-camera.jpg \
     --output overlay.jpg \
     --state-json board_state.json \
     --print-state \
     --device cpu          # use cuda:0 or mps if available
   ```
   Add `--real-data real` to mix labeled real images, `--force-train`/`--force-regen` to retrain/regenerate synth data. Weights are saved to `models/cell_grid.pt`.
3. Plan a move from the detected board (text-only LLM):
   ```bash
   cd mission2/code/tic-tac-toe-planner
   uv run tic_tac_toe_planner.py \
     --state-json ../tic_tac_toe_overlay/board_state.json \
     --model Qwen/Qwen3-4B-Instruct-2507
   ```
   Optional: `--obs-host localhost` to toggle OBS sources `1`–`9` in scene `シーン`.
4. Run the live pipeline to overlay the next action on the camera feed and stream to the virtual device:
   ```bash
   cd mission2/code/ttt-pipeline
   uv run python3 main.py \
     --source 0 \
     --weights ../tic_tac_toe_overlay/models/cell_grid.pt \
     --planner-model Qwen/Qwen3-4B-Instruct-2507 \
     --interval 5 \
     --display \
     --save-state latest_board.json
   ```
   The script polls the board every `--interval` seconds, calls the planner when the state changes, masks the chosen cell in red, and pipes frames to `/dev/video12` via `ffmpeg` (edit the hard-coded device in `main.py` if needed). Use `--display` to preview locally; press `q` to quit.

5. Copy Makefile to exector home. and run π0.5 inference
    ```
    make inference-pi
    ```

## Submission Details

### 1. Mission Description
- *Real world application of your mission*

### 2. Creativity
- *What is novel or unique in your approach?*
- *Innovation in design, methodology, or application*

### 3. Technical implementations
- *Teleoperation / Dataset capture*
    - Robotics ML Model's Task is "Pick up 3d printed figure and put on the target place".
    - Areas of Focus for Data Collection
        - Only see laptop rerun view (camera view) in order to get efficient information for robots.
        - Pick up and place Figure Carefully.
        - To avoid mixing learning data and reducing accuracy, move directly and accurately to the target area.
        - Add Target Bounding Box for Inference using OBS tool.
            <video controls width="360px">
                <source src="./media/TeleOperation/Teleop.mp4" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <img src="./media/TeleOperation/Rerun.PNG" style="margin-right: 10px; width: 360px;">
            <img src="./media/TeleOperation/OBS.PNG" style="margin-right: 10px; width: 360px;">
            <img src="./media/TeleOperation/TopView.PNG" style="margin-right: 10px; width: 360px;">
- *Training*
    - The powerful MI300X accelerator enabled fast training with larger batch sizes, significantly speeding up the model's learning process. 
- *Inference*
    - **Inference with Pi0.5 Model using Real Time Chunking (RTC)**
        - The Pi0.5 model, a lightweight policy network, utilizes Real Time Chunking (RTC) to predict and execute entire action sequences in a single inference step. This method, evolved from Action Chunking Transformers, processes movements as complete "chunks" (e.g., "grasp → lift → move → place") rather than frame-by-frame. This approach is particularly effective on AMD's high-performance hardware, where the parallel processing capabilities can be fully leveraged to minimize latency and ensure smooth, real-time robotic manipulation.
    - **Board State Detection using Fine-tuned YOLO**
        - A fine-tuned YOLOv8n model detects the contents and bounding box of each cell (empty, X, or O), forming the initial step of the inference pipeline. This output is then bifurcated: the natural language description of the board state is fed to the LLM for strategic planning, while the bounding box coordinates are used by the Pi0.5 model to guide the robot's physical movements.
    - **Reasoning with LLM**
        - Leveraging the powerful reasoning capabilities of a text-only Large Language Model (LLM), the system acts as a board game solver. It processes the YOLO-detected board state as text to determine and return the optimal next move.

### 4. Ease of use
- *How generalizable is your implementation across tasks or environments?*
- *Flexibility and adaptability of the solution*
- *Types of commands or interfaces needed to control the robot*

## Additional Links
*For example, you can provide links to:*

- *Link to a video of your robot performing the task*
- *URL of your dataset in Hugging Face*
    - https://huggingface.co/datasets/guren-kaina/tic-tac-toe-1
    - https://huggingface.co/datasets/guren-kaina/tic-tac-toe-2
    - https://huggingface.co/datasets/guren-kaina/tic-tac-toe-3
    - https://huggingface.co/datasets/guren-kaina/tic-tac-toe-4
- *URL of your model in Hugging Face*
    - https://huggingface.co/guren-kaina/tic-tac-toe-1
    - https://huggingface.co/guren-kaina/act-tic-tac-toe-2
    - https://huggingface.co/guren-kaina/pi05_tic-tac-toe-2
    - https://huggingface.co/guren-kaina/act-tic-tac-toe-3
    - https://huggingface.co/guren-kaina/pi05_tic-tac-toe-3
    - https://huggingface.co/guren-kaina/act-tic-tac-toe-4
    - https://huggingface.co/guren-kaina/pi05_tic-tac-toe-4
- *Link to a blog post describing your work*
    - soon

## Code submission

This is the directory tree of this repo, you need to fill in the `mission` directory with your submission details.

```terminal
AMD_Robotics_Hackathon_2025_ProjectTemplate-main/
├── README.md
└── mission
    ├── code
    │   └── <code and script>
    └── wandb
        └── <latest run directory copied from wandb of your training job>
```


The `latest-run` is generated by wandb for your training job. Please copy it into the wandb sub directory of you Hackathon Repo.

The whole dir of `latest-run` will look like below:

```terminal
$ tree outputs/train/smolvla_so101_2cube_30k_steps/wandb/
outputs/train/smolvla_so101_2cube_30k_steps/wandb/
├── debug-internal.log -> run-20251029_063411-tz1cpo59/logs/debug-internal.log
├── debug.log -> run-20251029_063411-tz1cpo59/logs/debug.log
├── latest-run -> run-20251029_063411-tz1cpo59
└── run-20251029_063411-tz1cpo59
    ├── files
    │   ├── config.yaml
    │   ├── output.log
    │   ├── requirements.txt
    │   ├── wandb-metadata.json
    │   └── wandb-summary.json
    ├── logs
    │   ├── debug-core.log -> /dataset/.cache/wandb/logs/core-debug-20251029_063411.log
    │   ├── debug-internal.log
    │   └── debug.log
    ├── run-tz1cpo59.wandb
    └── tmp
        └── code
```

**NOTES**

1. The `latest-run` is the soft link, please make sure to copy the real target directory it linked with all sub dirs and files.
2. Only provide (upload) the wandb of your last success pre-trained model for the Mission.
