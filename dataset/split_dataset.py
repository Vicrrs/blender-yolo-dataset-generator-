# ===============================
# split_dataset.py
# ===============================
"""
Gera dataset YOLO (train/val) **dentro da própria pasta** onde os scripts estão.

Uso típico (defaults adequados):

    cd /media/vicrrs/ARQUIVOS/blender/Data_Synthetic/SaoJorge/anotacoes
    python split_dataset.py          # cria ./dataset/
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Tuple, List

from PIL import Image
from tqdm import tqdm


def coco_to_yolo(box: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """[x, y, w, h]  -> [cx/img_w, cy/img_h, w/img_w, h/img_h]"""
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def collect_pairs(src: Path) -> List[Tuple[Path, Path]]:
    """
    Para cada *_coco.json procura a imagem correspondente
    (mesmo nome, porém sem o sufixo '_coco' e com extensão de imagem).
    """
    pairs = []
    for json_path in src.rglob("*_coco.json"):
        stem_no_coco = json_path.stem[:-5] if json_path.stem.endswith("_coco") else json_path.stem

        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            img_path = json_path.with_name(stem_no_coco + ext)
            if img_path.exists():
                pairs.append((img_path, json_path))
                break
    return pairs


def img_size(img_path: Path) -> Tuple[int, int]:
    """Abre a imagem só para pegar (w, h) – sem carregar pixels na RAM posteriormente."""
    with Image.open(img_path) as im:
        return im.width, im.height


def main(src: Path, dst: Path, split_ratio: float = 0.8, seed: int = 42):
    random.seed(seed)

    img_train = dst / "images" / "train"
    img_val   = dst / "images" / "val"
    lbl_train = dst / "labels" / "train"
    lbl_val   = dst / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    img_json_pairs = collect_pairs(src)
    n_pairs = len(img_json_pairs)
    print(f"Encontrados {n_pairs} pares (imagem + json).")

    if not n_pairs:
        raise SystemExit("[ERRO] Nenhum par encontrado — verifique o caminho --src")

    random.shuffle(img_json_pairs)
    split_idx   = int(n_pairs * split_ratio)
    train_pairs = img_json_pairs[:split_idx]
    val_pairs   = img_json_pairs[split_idx:]

    print(f"→  {len(train_pairs)} treino   |   {len(val_pairs)} validação")

    for split_name, pairs, img_out, lbl_out in (
        ("train", train_pairs, img_train, lbl_train),
        ("val",   val_pairs,   img_val,  lbl_val),
    ):
        for img_path, json_path in tqdm(pairs, desc=f"Processando {split_name}"):
            # 1) copiar imagem
            shutil.copy2(img_path, img_out / img_path.name)

            # 2) ler JSON
            with open(json_path, "r") as f:
                coco = json.load(f)

            # 3) obter tamanho da imagem
            if "images" in coco and coco["images"]:
                # JSON realmente no formato COCO
                img_info = coco["images"][0]
                img_w, img_h = img_info["width"], img_info["height"]
                img_id       = img_info.get("id", 0)
                anns = [ann for ann in coco["annotations"] if ann.get("image_id") == img_id]
            else:
                # JSON gerado pelo seu script Blender (apenas 'annotations')
                img_w, img_h = img_size(img_path)
                anns = coco["annotations"]

            # 4) converter bboxes
            lines = []
            for ann in anns:
                bbox = ann["bbox"]                       # [x, y, w, h]
                bbox_yolo = coco_to_yolo(bbox, img_w, img_h)
                class_id = 0                             # só existe a classe SãoJorge
                lines.append(" ".join([str(class_id)] + [f"{v:.6f}" for v in bbox_yolo]))

            # 5) gravar txt YOLO
            label_file = lbl_out / f"{img_path.stem}.txt"
            label_file.write_text("\n".join(lines))


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent          # /anotacoes

    parser = argparse.ArgumentParser(
        description="Converte COCO JSON em YOLO e divide em train/val (tudo dentro da pasta atual)."
    )
    parser.add_argument("--src",   default=str(base_dir),            help="Diretório com subpastas anotacao-*")
    parser.add_argument("--dst",   default=str(base_dir / "dataset"), help="Diretório de saída para dataset YOLO")
    parser.add_argument("--split", type=float, default=0.8,          help="Proporção para treino (0-1)")
    parser.add_argument("--seed",  type=int,   default=42,           help="Seed para shuffle")
    args = parser.parse_args()

    main(Path(args.src), Path(args.dst), args.split, args.seed)
