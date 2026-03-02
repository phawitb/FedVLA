import os
import random
import numpy as np
import metaworld
import mujoco
import importlib.util
import metaworld.policies as mp
from datetime import datetime

TASKS = ["door-lock-v3", "drawer-close-v3", "window-open-v3"]
NUM_TRAJ = 40
STEPS_PER_TRAJ = 150
CAMERA = "corner2"
H, W = 480, 480
OUT_ROOT = "dataset"
SEED = 0

def load_policy(task_name: str):
    policy_dir = os.path.dirname(mp.__file__)
    search_term = task_name.split("-v")[0].replace("-", "_")
    target_file = next((f for f in os.listdir(policy_dir) if search_term in f and f.endswith(".py")), None)
    if target_file is None:
        raise FileNotFoundError(f"Policy file not found for task: {task_name}")

    spec = importlib.util.spec_from_file_location("pol", os.path.join(policy_dir, target_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cls_name = f"Sawyer{''.join([x.capitalize() for x in task_name.split('-')])}Policy"
    if not hasattr(mod, cls_name):
        raise AttributeError(f"Policy class not found: {cls_name}")

    return getattr(mod, cls_name)()

def collect_one_task(task_name: str, timestamp: str):
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    policy = load_policy(task_name)
    renderer = mujoco.Renderer(env.model, height=H, width=W)

    images, states, actions = [], [], []

    for _ in range(NUM_TRAJ):
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        obs, _ = env.reset()

        for _ in range(STEPS_PER_TRAJ):
            renderer.update_scene(env.data, camera=CAMERA)
            frame = renderer.render()

            act = policy.get_action(obs)
            act = np.clip(act, -1.0, 1.0).astype(np.float32)

            images.append(frame)
            states.append(obs)
            actions.append(act)

            obs, _, _, _, _ = env.step(act)

    save_dir = os.path.join(OUT_ROOT, f"{task_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(save_dir, "dataset.npz"),
        images=np.array(images, dtype=np.uint8),
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        task_name=np.array([task_name]),
        num_trajectories=np.array([NUM_TRAJ]),
        steps_per_traj=np.array([STEPS_PER_TRAJ]),
        camera_name=np.array([CAMERA]),
        seed=np.array([SEED]),
    )

    renderer.close()
    env.close()
    return save_dir

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for t in TASKS:
        try:
            out_dir = collect_one_task(t, timestamp)
            print("saved:", out_dir)
        except Exception as e:
            print(f"{t} error: {e}")

if __name__ == "__main__":
    main()