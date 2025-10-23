import os
import sys
import argparse
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

for sub in ["models", "models/Faster_RCNN", "models/YOLO", "exploratory_data_analysis"]:
    full_path = os.path.join(ROOT_DIR, sub)
    if os.path.isdir(full_path) and full_path not in sys.path:
        sys.path.append(full_path)

def run_in_dir(cmd, rel_dir):
    target_dir = os.path.join(ROOT_DIR, rel_dir)
    print(f"\n📁 Changing directory to: {target_dir}")
    os.chdir(target_dir)
    print(f"🚀 Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDD100K Applied CV Pipeline Runner")
    parser.add_argument(
        "--task",
        choices=["eda", "frcnn", "yolo"],
        required=True,
        help="Choose a pipeline to run: eda | frcnn | yolo",
    )
    args = parser.parse_args()

    os.environ["PYTHONPATH"] = ROOT_DIR

    if args.task == "eda":
        run_in_dir("streamlit run main.py", "exploratory_data_analysis")

    elif args.task == "frcnn":
        run_in_dir("python eval_frcnn.py", "models/Faster_RCNN")

    elif args.task == "yolo":
        run_in_dir("python eval_yolo.py", "models/YOLO")
