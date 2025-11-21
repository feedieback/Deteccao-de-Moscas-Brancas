# -*- coding: utf-8 -*-
"""
Whitefly Auto-Labeler V3
Melhorias:
 - Pré-processamento com CLAHE e denoise
 - Detecção multi-faixa HSV (adulto, ninfa, fallback)
 - Combinação máscara_cor + bordas (Canny) + filtragem por forma
 - Score multi-critério e NMS simples
 - Gera arquivos .txt no formato YOLO (classe 0)
 - Visualizações de amostras e relatório simples
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ---------- Ajuste de parâmetros principais ----------
DEFAULT_CONFIDENCE = 0.55  # Limiar para salvar detections (pode reduzir para mais labels)
VIS_SAMPLES_PER_SPLIT = 12
# ----------------------------------------------------

class WhiteflyAutoLabelerV3:
    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE, debug: bool = False):
        self.confidence_threshold = float(confidence_threshold)
        self.debug = debug

        # Faixas HSV calibradas a partir das suas imagens
        self.stages = {
            'adulto': {
                'hsv_lower': np.array([18, 20, 150], dtype=np.uint8),
                'hsv_upper': np.array([42, 160, 255], dtype=np.uint8),
                'min_area': 30,
                'max_area': 3000,
                'weight': 1.0
            },
            'ninfa': {
                'hsv_lower': np.array([22, 60, 120], dtype=np.uint8),
                'hsv_upper': np.array([65, 255, 255], dtype=np.uint8),
                'min_area': 10,
                'max_area': 1600,
                'weight': 0.95
            },
            'geral': {
                'hsv_lower': np.array([0, 0, 180], dtype=np.uint8),
                'hsv_upper': np.array([85, 255, 255], dtype=np.uint8),
                'min_area': 8,
                'max_area': 4000,
                'weight': 0.8
            }
        }

        # Faixa da folha (contexto)
        self.leaf_hsv_lower = np.array([30, 25, 30], dtype=np.uint8)
        self.leaf_hsv_upper = np.array([90, 255, 255], dtype=np.uint8)

        print("="*70)
        print("WHITEFLY AUTO-LABELER V3")
        print("="*70)
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Debug: {self.debug}")
        print("Stages:", ", ".join(self.stages.keys()))

    # ---------- Preprocess ----------
    def _preprocess(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        # Resize small images moderately to stabilize processing
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) < 600:
            scale = 1.5
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

        # CLAHE on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Denoise and light blur
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 9, 9, 7, 21)
        blurred = cv2.GaussianBlur(denoised, (3,3), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        return {
            'orig': img,
            'enhanced': enhanced,
            'denoised': denoised,
            'blurred': blurred,
            'hsv': hsv,
            'gray': gray,
            'edges': edges,
            'scale': scale
        }

    # ---------- Create combined mask for a stage ----------
    def _stage_mask(self, hsv: np.ndarray, stage_name: str) -> np.ndarray:
        s = self.stages[stage_name]
        mask = cv2.inRange(hsv, s['hsv_lower'], s['hsv_upper'])
        # small morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    # ---------- Candidate extraction ----------
    def _extract_candidates(self, pre: Dict[str, np.ndarray]) -> List[Dict]:
        hsv = pre['hsv']
        edges = pre['edges']
        img_h, img_w = pre['orig'].shape[:2]

        candidates = []
        for stage_name in self.stages.keys():
            mask = self._stage_mask(hsv, stage_name)

            # Combine with edges: require some edge overlap to reduce background blobs
            edges_dil = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            mask_edges = cv2.bitwise_and(mask, mask, mask=edges_dil)

            # Fuse masks: keep both pure-color blobs and color+edge blobs
            fused = cv2.bitwise_or(mask, mask_edges)

            # Find contours on fused
            cnts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sconf = self.stages[stage_name]
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < sconf['min_area'] or area > sconf['max_area']:
                    continue

                x,y,w,h = cv2.boundingRect(cnt)
                aspect = w / (h + 1e-6)
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4*np.pi*area / (perimeter*perimeter)) if perimeter > 0 else 0
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull) if hull is not None else 0
                solidity = area / (hull_area + 1e-6)
                extent = area / (w*h + 1e-6)
                hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
                hu_score = 1.0/(1.0 + np.abs(hu[0]))  # weak indicator

                candidate = {
                    'stage': stage_name,
                    'cnt': cnt,
                    'bbox': [x,y,w,h],
                    'area': area,
                    'aspect': aspect,
                    'circularity': circularity,
                    'solidity': solidity,
                    'extent': extent,
                    'hu0': float(hu[0]),
                    'hu_score': float(hu_score),
                    'mask_mean': float(np.mean(mask[y:y+h, x:x+w]) / 255.0),
                    'fused_mean': float(np.mean(fused[y:y+h, x:x+w]) / 255.0)
                }
                candidates.append(candidate)

        return candidates

    # ---------- Confidence scoring ----------
    def _score_candidate(self, cand: Dict, pre: Dict[str, np.ndarray]) -> float:
        sconf = self.stages[cand['stage']]

        score = 0.0
        # area closeness
        ideal = (sconf['min_area'] + sconf['max_area'])/2.0
        area_score = max(0.0, 1.0 - abs(cand['area'] - ideal)/ideal)
        score += area_score * 0.30

        # solidity (solid insect body)
        score += min(1.0, cand['solidity']) * 0.18

        # circularity (but whitefly not perfectly circular)
        score += min(1.0, cand['circularity']*2.0) * 0.15

        # mask presence (how filled mask is)
        score += cand['mask_mean'] * 0.12

        # fused_mean (color+edge)
        score += cand['fused_mean'] * 0.10

        # aspect close to expected (prefer slightly oval)
        aspect_ideal = 1.2
        aspect_score = max(0.0, 1.0 - abs(cand['aspect'] - aspect_ideal)/aspect_ideal)
        score += aspect_score * 0.10

        # hu score small boost for compact shapes
        score += min(1.0, cand['hu_score']) * 0.05

        # stage weight
        score *= sconf.get('weight', 1.0)

        return float(max(0.0, min(1.0, score)))

    # ---------- Non max suppression (IoU on bboxes) ----------
    def _nms(self, detections: List[Dict], iou_thres: float = 0.35) -> List[Dict]:
        if not detections:
            return []
        dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []
        for d in dets:
            x1,y1,w1,h1 = d['bbox']
            x1b,y1b,x1e,y1e = x1, y1, x1+w1, y1+h1
            add = True
            for k in keep:
                x2,y2,w2,h2 = k['bbox']
                x2b,y2b,x2e,y2e = x2, y2, x2+w2, y2+h2
                xx1 = max(x1b, x2b)
                yy1 = max(y1b, y2b)
                xx2 = min(x1e, x2e)
                yy2 = min(y1e, y2e)
                if xx2 > xx1 and yy2 > yy1:
                    inter = (xx2-xx1)*(yy2-yy1)
                    union = w1*h1 + w2*h2 - inter
                    iou = inter / (union + 1e-6)
                    if iou > iou_thres:
                        add = False
                        break
            if add:
                keep.append(d)
        return keep

    # ---------- Main detect on single image ----------
    def detect_on_image(self, image_path: Path) -> List[Dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        pre = self._preprocess(img)
        candidates = self._extract_candidates(pre)
        detections = []
        for c in candidates:
            score = self._score_candidate(c, pre)
            if score >= (self.confidence_threshold * 0.4):  # keep looser then final nms
                det = {
                    'bbox': c['bbox'],
                    'score': score,
                    'stage': c['stage']
                }
                detections.append(det)

        # NMS and final filter
        detections = self._nms(detections, iou_thres=0.35)
        final = [d for d in detections if d['score'] >= self.confidence_threshold]

        # scale back if preprocessing resized image
        scale = pre.get('scale', 1.0)
        if scale != 1.0:
            for d in final:
                x,y,w,h = d['bbox']
                d['bbox'] = [int(x/scale), int(y/scale), int(w/scale), int(h/scale)]

        return final

    # ---------- Convert to YOLO format ----------
    def to_yolo_lines(self, dets: List[Dict], img_w: int, img_h: int) -> List[str]:
        lines = []
        for d in dets:
            x,y,w,h = d['bbox']
            cx = (x + w/2) / img_w
            cy = (y + h/2) / img_h
            ww = w / img_w
            hh = h / img_h
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            ww = float(np.clip(ww, 0.0, 1.0))
            hh = float(np.clip(hh, 0.0, 1.0))
            lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
        return lines

    # ---------- Visualization ----------
    def visualize(self, src_img_path: Path, detections: List[Dict], out_path: Path):
        img = cv2.imread(str(src_img_path))
        if img is None:
            return
        for d in detections:
            x,y,w,h = d['bbox']
            s = d['score']
            color = (0,255,0) if s>=0.8 else (0,255,255) if s>=0.6 else (0,165,255)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{s:.2f}", (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

    # ---------- Process entire dataset ----------
    def process_dataset(self, dataset_root: Path, visualize_samples: int = VIS_SAMPLES_PER_SPLIT):
        dataset_root = Path(dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        # backup existing labels
        labels_dir = dataset_root / 'labels'
        if labels_dir.exists():
            backup_dir = dataset_root / f"labels_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(labels_dir, backup_dir)
            if self.debug:
                print("Backup of labels created at:", backup_dir)

        report = {
            'total_images': 0,
            'images_with_detections': 0,
            'total_detections': 0,
            'by_split': {}
        }

        vis_dir = dataset_root / 'auto_labels_v3_visual'
        vis_dir.mkdir(exist_ok=True)

        for split in ['train','val','test']:
            imgs = list((dataset_root/'images'/split).glob('*.jpg')) + list((dataset_root/'images'/split).glob('*.png'))
            (dataset_root/'labels'/split).mkdir(parents=True, exist_ok=True)
            split_stats = {'images': len(imgs), 'detections': 0, 'images_with_detections': 0}
            sample_vis = 0
            print(f"\nProcessing {split}: {len(imgs)} images")
            for img_path in tqdm(imgs, desc=f"{split}"):
                report['total_images'] += 1
                dets = self.detect_on_image(img_path)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h,w = img.shape[:2]
                yolo_lines = self.to_yolo_lines(dets, w, h)
                label_file = dataset_root/'labels'/split/f"{img_path.stem}.txt"
                label_file.write_text("\n".join(yolo_lines), encoding='utf-8')  # empty ok
                if dets:
                    report['images_with_detections'] += 1
                    split_stats['images_with_detections'] += 1
                report['total_detections'] += len(dets)
                split_stats['detections'] += len(dets)
                # visualize a few
                if sample_vis < visualize_samples and dets:
                    self.visualize(img_path, dets, vis_dir / f"{split}_{img_path.name}")
                    sample_vis += 1
            report['by_split'][split] = split_stats

        # save report
        report_file = dataset_root / f'auto_label_v3_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_file.write_text(json.dumps(report, indent=2))
        print("\nAuto-labeling complete. Report:", report_file)
        return report

# -------------------- Script usage --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Whitefly Auto-Labeler V3")
    parser.add_argument("--dataset", type=str, required=False,
                        default=r"C:\Users\Victor\Documents\TCC\IA\datasets\ip102_yolo_white_fly",
                        help="dataset root (images/, labels/)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="confidence threshold")
    parser.add_argument("--vis", type=int, default=VIS_SAMPLES_PER_SPLIT, help="visual samples per split")
    parser.add_argument("--debug", action='store_true', help="debug prints")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print("Dataset not found:", root)
        exit(1)

    labeler = WhiteflyAutoLabelerV3(confidence_threshold=args.conf, debug=args.debug)
    report = labeler.process_dataset(root, visualize_samples=args.vis)
    print("\nSummary:")
    print(json.dumps(report, indent=2))
    print("\nDone.")
