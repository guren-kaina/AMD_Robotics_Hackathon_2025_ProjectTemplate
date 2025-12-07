CONDA_ENV = lerobot

.PHONY: port
port:
	conda run -n $(CONDA_ENV) --no-capture-output lerobot-find-port

.PHONY: camera
camera:
	conda run -n $(CONDA_ENV) lerobot-find-cameras opencv 

.PHONY: teleop
teleop:
	conda run -n $(CONDA_ENV) lerobot-teleoperate \
		--robot.type=so101_follower \
		--robot.port=/dev/follower \
		--robot.id=kaina_follower_arm \
		--teleop.type=so101_leader \
		--teleop.port=/dev/leader \
		--teleop.id=kaina_leader_arm

.PHONY: teleop-with-camera
teleop-with-camera:
	conda run -n $(CONDA_ENV) lerobot-teleoperate \
		--robot.type=so101_follower \
		--robot.port=/dev/follower \
		--robot.id=kaina_follower_arm \
		--robot.cameras="{top: {type: opencv, index_or_path: '/dev/video12', width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: '/dev/side-camera-1', width: 640, height: 480, fps: 30}, side2: {type: opencv, index_or_path: '/dev/side-camera-2', width: 640, height: 480, fps: 30}, glipper: {type: opencv, index_or_path: '/dev/arm-camera', width: 640, height: 480, fps: 30}}" \
		--teleop.type=so101_leader \
		--teleop.port=/dev/leader \
		--teleop.id=kaina_leader_arm \
		--display_data=true

.PHONY: record
record:
	conda run -n $(CONDA_ENV) lerobot-record \
		--robot.type=so101_follower \
		--robot.port=/dev/follower \
		--robot.id=kaina_follower_arm \
		--robot.cameras="{top: {type: opencv, index_or_path: '/dev/video12', width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: '/dev/side-camera-1', width: 640, height: 480, fps: 30}, side2: {type: opencv, index_or_path: '/dev/side-camera-2', width: 640, height: 480, fps: 30}, glipper: {type: opencv, index_or_path: '/dev/arm-camera', width: 640, height: 480, fps: 30}}" \
		--teleop.type=so101_leader \
		--teleop.port=/dev/leader \
		--teleop.id=kaina_leader_arm \
		--display_data=true \
		--dataset.repo_id=guren-kaina/tic-tac-toe-3 \
		--dataset.num_episodes=45 \
		--dataset.episode_time_s=30 \
		--dataset.reset_time_s=10 \
		--dataset.single_task="pickup figure from yellow to red" \
		--dataset.root=$$HOME/so101_dataset/$(shell date +'%Y%m%d-%H%M%S')/ \
		--dataset.push_to_hub=false \
		--display_data=true

.PHONY: inference-act
inference-act:
	conda run -n $(CONDA_ENV) lerobot-record \
		--robot.type=so101_follower \
		--robot.port=/dev/follower \
		--robot.id=kaina_follower_arm \
		--robot.cameras="{top: {type: opencv, index_or_path: '/dev/video12', width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: '/dev/side-camera-1', width: 640, height: 480, fps: 30}, side2: {type: opencv, index_or_path: '/dev/side-camera-2', width: 640, height: 480, fps: 30}, glipper: {type: opencv, index_or_path: '/dev/arm-camera', width: 640, height: 480, fps: 30}}" \
		--dataset.single_task="Pick cube from source position and stack it on the fixed cube at target position" \
		--dataset.root=$$PWD/eval_lerobot_dataset/$(shell date +'%Y%m%d-%H%M%S')/ \
		--dataset.episode_time_s=20 \
		--dataset.num_episodes=1 \
        --dataset.repo_id=guren-kaina/eval_act-tic-tac-toe-3 \
		--policy.path=guren-kaina/act-tic-tac-toe-3 \
		--dataset.push_to_hub=false \
        --display_data=true
	
.PHONY: inference-pi
inference-pi:
	conda run -n $(CONDA_ENV) --no-capture-output python lerobot/examples/rtc/eval_with_real_robot.py \
		--policy.path=guren-kaina/pi05_tic-tac-toe-3 \
		--policy.device=cuda \
		--rtc.enabled=true \
		--rtc.execution_horizon=20 \
		--robot.type=so101_follower \
		--robot.port=/dev/follower \
		--robot.id=kaina_follower_arm \
		--robot.cameras="{top: {type: opencv, index_or_path: '/dev/video12', width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: '/dev/side-camera-1', width: 640, height: 480, fps: 30}, side2: {type: opencv, index_or_path: '/dev/side-camera-2', width: 640, height: 480, fps: 30}, glipper: {type: opencv, index_or_path: '/dev/arm-camera', width: 640, height: 480, fps: 30}}"  \
		--task="pickup figure from yellow to red" \
		--duration=240
