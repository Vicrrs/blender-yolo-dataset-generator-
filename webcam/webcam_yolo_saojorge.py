#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================= #
# webcam_yolo_saojorge.py                       #
# ============================================= #
"""
Detecta o objeto “SaoJorge” em tempo-real usando a webcam
e o modelo YOLOv8 treinado com train_yolo.py.

Uso básico (GPU 0):

    python webcam_yolo_saojorge.py --weights runs/train-saojorge/exp/weights/best.pt --device 0

Para CPU:

    python webcam_yolo_saojorge.py --device cpu
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main(weights: Path, device: str, cam_index: int, imgsz: int, conf: float):
    model = YOLO(str(weights))
    if device:
        model.to(device)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"❌ Não consegui abrir a câmera {cam_index}")

    print("✅ Webcam aberta. Pressione ‘q’ para sair…")
    names = model.names
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Falha ao ler frame, tentando próximo…")
            continue

        results = model(frame, imgsz=imgsz, conf=conf)   # lista [Results]
        annotated = results[0].plot(labels=True, boxes=True)  # desenha boxes

        cv2.imshow("YOLOv8", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam inference com YOLOv8")
    parser.add_argument("--weights", default="runs/train-saojorge/exp/weights/best.pt",
                        help="Arquivo .pt treinado (best.pt)")
    parser.add_argument("--device",  default="", help="0,1,… ou 'cpu'")
    parser.add_argument("--cam",     type=int, default=0, help="Índice da câmera (0 padrão)")
    parser.add_argument("--imgsz",   type=int, default=640, help="Tamanho de entrada da rede")
    parser.add_argument("--conf",    type=float, default=0.25, help="Conf. mínima para mostrar bbox")
    args = parser.parse_args()

    main(Path(args.weights), args.device, args.cam, args.imgsz, args.conf)

