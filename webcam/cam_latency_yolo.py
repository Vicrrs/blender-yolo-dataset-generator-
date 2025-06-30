#!/usr/bin/env python3
# cam_latency_yolo.py  (versão com visualização + latência)
# ------------------------------------------------------------------
# • Mede o tempo até a 1ª detecção (> limiar) em até 3 webcams.
# • Exibe em tempo real as predições em janelas OpenCV.
#
# Uso:
#   python cam_latency_yolo.py \
#       --weights /caminho/best.pt \
#       --device cuda:0
#
# Saída ao encerrar (tecla **q**):
#   =========== LATÊNCIA - 1ª detecção ==========
#   Camera 0 : 112.3 ms
#   Camera 2 :  95.8 ms
#   Camera 4 : 108.7 ms
#   Média            : 105.6 ms
#   =============================================
#
# Requisitos:
#   pip install -U ultralytics opencv-python
# ------------------------------------------------------------------
import argparse, cv2, sys, time
from pathlib import Path
from ultralytics import YOLO
import torch


def open_cameras(n=3, max_test=10):
    """Abre até n webcams válidas (0..max_test-1) e retorna listas (caps, idxs)."""
    caps, idxs = [], []
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if ok:
            caps.append(cap)
            idxs.append(idx)
            if len(caps) == n:
                break
        else:
            cap.release()
    return caps, idxs


def main(weights, device, imgsz, conf, maxcams):
    # 1. Modelo
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Carregando modelo em {device} …")
    model = YOLO(str(weights))
    model.to(device)
    model.fuse()

    # 2. Webcams
    caps, idxs = open_cameras(maxcams)
    if not caps:
        sys.exit("[ERRO] Nenhuma webcam encontrada.")
    print(f"[INFO] Testando câmeras {idxs}  (limiar conf. {conf})")

    win_names = [f"cam{idx}" for idx in idxs]
    for name in win_names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    # 3. Medir latência
    t0 = {i: None for i in range(len(caps))}
    first_hit = {}

    try:
        while True:
            frames = []
            alive = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    continue
                frames.append(frame)
                alive.append(i)
                if t0[i] is None:
                    t0[i] = time.time()

            if not frames:
                break

            results = model.predict(
                frames,
                imgsz=imgsz,
                conf=conf,
                device=device,
                half=device.startswith("cuda"),
                verbose=False,
            )

            for k, res in enumerate(results):
                cam_idx = alive[k]
                im = res.plot()          # BGR com caixas
                cv2.imshow(win_names[cam_idx], im)

                # registra 1ª detecção
                if (cam_idx not in first_hit) and res.boxes.shape[0]:
                    first_hit[cam_idx] = (time.time() - t0[cam_idx]) * 1000  # ms

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # encerra automaticamente após medir todas
            if len(first_hit) == len(caps):
                if cv2.waitKey(500) & 0xFF == ord("q"):  # espera 0,5 s para ver as caixas
                    break

    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

    # 4. Relatório
    print("\n=========== LATÊNCIA - 1ª detecção ==========")
    for cam_local_idx, ms in first_hit.items():
        print(f"Camera {idxs[cam_local_idx]} : {ms:.1f} ms")
    if first_hit:
        print(f"Média            : {sum(first_hit.values())/len(first_hit):.1f} ms")
    else:
        print("Nenhuma detecção registrada.")
    print("=============================================")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    default_weights = base / "runs/train-saojorge/train/weights/best.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(default_weights))
    ap.add_argument("--device",  default="")
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--conf",    type=float, default=0.25)
    ap.add_argument("--maxcams", type=int,  default=3, help="quantas webcams abrir")
    args = ap.parse_args()

    main(Path(args.weights), args.device, args.imgsz, args.conf, args.maxcams)
