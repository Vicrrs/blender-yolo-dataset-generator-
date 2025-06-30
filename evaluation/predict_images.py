# ===============================
# predict_images.py (v2)
# ===============================
"""
Mostra, uma a uma, as predições YOLOv8 sobre todas as imagens de um diretório
usando Matplotlib.


python predict_images.py --dir /outro/pasta --weights /caminho/meu_best.pt --device cuda:0
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def get_image_paths(dir_path: Path):
    """Retorna lista ordenada de imagens em *dir_path* (busca recursiva)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".HEIC"}
    return [p for p in sorted(dir_path.rglob("*")) if p.suffix.lower() in exts]


def display_prediction(model: YOLO, img_path: Path, imgsz: int, conf: float):
    """Roda predição e plota resultado com Matplotlib."""
    results = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False, show=False)
    im_bgr = results[0].plot()  # resultado desenhado (BGR)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(img_path.name)
    plt.imshow(im_rgb)
    plt.show()



def main(dir_path: Path, weights: Path, device: str, imgsz: int, conf: float):
    model = YOLO(str(weights))
    if device:
        model.to(device)

    img_paths = get_image_paths(dir_path)
    if not img_paths:
        raise SystemExit(f"[ERRO] Nenhuma imagem encontrada em {dir_path}")

    print(f"Encontradas {len(img_paths)} imagens em {dir_path}. Pressione <Enter> para avançar, 'q' para sair.")

    for img in img_paths:
        user_in = input(f"\nPróxima imagem: {img.name}  →  <Enter> / 'q': ").strip().lower()
        if user_in == "q":
            break
        display_prediction(model, img, imgsz, conf)


if __name__ == "__main__":
    base_dir = Path("/media/vicrrs/ARQUIVOS/blender/Data_Synthetic/SaoJorge/imgs/imagens/Sj")
    default_weights = Path(__file__).resolve().parent / "runs/train-saojorge/train/weights/best.pt"

    parser = argparse.ArgumentParser(description="YOLOv8 batch prediction + Matplotlib.")
    parser.add_argument("--dir",     default=str(base_dir),          help="Diretório com imagens")
    parser.add_argument("--weights", default=str(default_weights),   help="Checkpoint .pt")
    parser.add_argument("--device",  default="",                   help="cuda:0, cpu, etc.")
    parser.add_argument("--imgsz",   type=int,   default=640,        help="Tamanho (resize) enviado à rede")
    parser.add_argument("--conf",    type=float, default=0.25,       help="Confiança mínima")
    args = parser.parse_args()

    main(Path(args.dir), Path(args.weights), args.device, args.imgsz, args.conf)