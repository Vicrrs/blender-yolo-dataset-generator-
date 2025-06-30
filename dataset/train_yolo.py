# =============================== #
#  train_yolo.py (versão final)   #
# =============================== #
"""
Treina YOLOv8 no dataset criado por split_dataset.py.

Uso básico (GPU 0):

    cd /media/vicrrs/ARQUIVOS/blender/Data_Synthetic/SaoJorge/anotacoes
    python train_yolo.py --device 0
"""

import argparse, subprocess, yaml
from pathlib import Path


def create_data_yaml(dataset_dir: Path, yaml_out: Path):
    cfg = {
        "path":  str(dataset_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "names": {0: "SaoJorge"},
    }
    yaml_out.write_text(yaml.safe_dump(cfg, sort_keys=False))


def main(dataset_dir: Path, model: str, epochs: int, imgsz: int,
         device: str, workers: int, project: str):
    dataset_dir = dataset_dir.resolve()
    yaml_file   = dataset_dir.parent / "saojorge.yaml"
    create_data_yaml(dataset_dir, yaml_file)

    cmd = [
        "yolo", "detect", "train",
        f"model={model}",
        f"data={yaml_file}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"workers={workers}",
        f"project={project}",
    ]
    if device:
        cmd.append(f"device={device}")

    print("\n[Ultralytics] Comando:\n  ", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=str(here / "dataset"))
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--device", default="")         # "0" ou "cpu"
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/train-saojorge")
    args = parser.parse_args()

    main(Path(args.data), args.model, args.epochs, args.imgsz,
         args.device, args.workers, args.project)
