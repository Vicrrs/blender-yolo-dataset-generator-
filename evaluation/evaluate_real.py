#!/usr/bin/env python3
# evaluate_real.py  ---------------------------------------------------------
# TESTE RÁPIDO ─ verifica se o modelo YOLOv8 detecta o objeto em fotos REAIS
# (não precisa de labels). Aceita JPG/PNG e HEIC/HEIF.
#
# Uso:
#   python evaluate_real.py --images /pasta/fotos --weights modelo.pt
#
# Saída:
#   • total de imagens analisadas
#   • quantas tiveram ≥1 detecção acima do threshold
#   • taxa de cobertura (%)
#   • média / min / máx da confiança da melhor caixa por imagem
#   • opcional: salva N exemplos com as caixas (runs/predict-*)
#
# Requisitos:
#   pip install -U ultralytics pillow-heif
# --------------------------------------------------------------------------
import argparse, sys, numpy as np
from pathlib import Path
from ultralytics import YOLO


# ──────────────── abre JPG/PNG direto e HEIC/HEIF via pillow-heif ───────────
def load_image_any(path: Path):
    ext = path.suffix.lower()
    if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
        return str(path)  # YOLO lê via cv2
    if ext in {'.heic', '.heif'}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            from PIL import Image
            return np.array(Image.open(path).convert('RGB'))
        except ImportError:
            print('[AVISO] pillow-heif não instalado — pulei', path.name)
            return None
    return None


def main():
    ap = argparse.ArgumentParser('Quick check YOLOv8 em imagens reais.')
    ap.add_argument('--images',  required=True, help='Diretório com fotos reais')
    ap.add_argument('--weights', required=True, help='Modelo .pt')
    ap.add_argument('--device',  default='cuda:0', help='cuda:0 | cpu | ""')
    ap.add_argument('--conf',    type=float, default=0.25, help='Threshold')
    ap.add_argument('--imgsz',   type=int,   default=640)
    ap.add_argument('--samples', type=int,   default=12,
                    help='Salva até N exemplos com caixas (0 = não salvar)')
    args = ap.parse_args()

    img_dir = Path(args.images)
    files = sorted([p for p in img_dir.rglob('*')
                    if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif',
                                            '.tiff','.webp','.heic','.heif'}])
    if not files:
        sys.exit('[ERRO] Nenhuma imagem encontrada.')

    model = YOLO(args.weights)
    if args.device:
        model.to(args.device)

    total, ok, best_confs = 0, 0, []
    save_dir = None
    for path in files:
        img = load_image_any(path)
        if img is None:
            continue
        res = model.predict(
            source=img,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            save=(args.samples and ok < args.samples),
            save_txt=False
        )
        dets = res[0].boxes
        total += 1
        if dets.shape[0]:
            ok += 1
            best_confs.append(float(dets.conf.max()))
            if save_dir is None:
                save_dir = res[0].save_dir

    print('\n=========== RELATÓRIO RÁPIDO ===========')
    print(f'Imagens analisadas        : {total}')
    print(f'Com ≥1 detecção (> {args.conf}) : {ok}')
    print(f'Taxa de cobertura         : {ok/total*100:.1f} %')
    if best_confs:
        print(f'Conf. média (top-1)       : {np.mean(best_confs):.3f}')
        print(f'Conf. mínima (top-1)      : {np.min(best_confs):.3f}')
        print(f'Conf. máxima (top-1)      : {np.max(best_confs):.3f}')
    else:
        print('Nenhuma detecção acima do limiar.')
    if save_dir:
        print(f'Exemplos salvos em        : {save_dir}')
    print('========================================\n')


if __name__ == '__main__':
    main()
