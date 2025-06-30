#!/usr/bin/env python3
"""
Detecta o objeto São Jorge em TODAS as webcams encontradas.

Uso rápido (defaults já corretos):
    python multi_webcam_yolo.py

Params opcionais:
    --weights  caminho p/ pesos .pt       (default: runs/train-saojorge/train/weights/best.pt)
    --device   'cuda:0', 'cpu', ''        (default: cuda:0 se houver)
    --imgsz    lado da imagem quadrada    (default: 640)
    --conf     confiança mínima           (default: 0.25)
    --maxcams  quantos índices testar     (default: 10)
"""

import argparse
import cv2
import sys
import time
from pathlib import Path

from ultralytics import YOLO
import torch


def find_cameras(max_test: int = 10):
    """Retorna lista de índices de câmera que abriram com sucesso."""
    cams = []
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap is None or not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            cams.append(cap)
        else:
            cap.release()
    return cams


def main(weights, device, imgsz, conf, maxcams):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Carregando modelo em {device} …")
    model = YOLO(str(weights))
    model.to(device)
    model.fuse()

    caps = find_cameras(maxcams)
    if not caps:
        sys.exit("[ERRO] Nenhuma webcam encontrada :(")
    print(f"[INFO] {len(caps)} webcams abertas:", [int(c.get(cv2.CAP_PROP_POS_FRAMES)) for c in caps])  # só p/ mostrar índices

    win_names = [f"cam{idx}" for idx in range(len(caps))]
    for name in win_names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = []
            to_drop = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    to_drop.append(i)
                    continue
                frames.append(frame)

            for i in sorted(to_drop, reverse=True):
                caps[i].release()
                caps.pop(i)
                cv2.destroyWindow(win_names[i])
                win_names.pop(i)

            if not caps:
                break

            results = model.predict(
                frames,
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
                half= (device.startswith("cuda"))
            )

            for frame_idx, res in enumerate(results):
                im = res.plot()                # BGR
                cv2.imshow(win_names[frame_idx], im)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    default_weights = base / "runs/train-saojorge/train/weights/best.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(default_weights))
    ap.add_argument("--device",  default="")
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--conf",    type=float, default=0.25)
    ap.add_argument("--maxcams", type=int, default=10,
                    help="quantos índices 0..N testar ao procurar webcams")
    args = ap.parse_args()

    main(Path(args.weights), args.device, args.imgsz, args.conf, args.maxcams)
