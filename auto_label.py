# -*- coding: utf-8 -*-
"""
Whitefly Auto-Labeler V3
------------------------
Sistema de visão computacional clássica (não-IA) para geração automática de 
anotações (labels) para treinamento de modelos YOLO.

Objetivo:
    Identificar moscas-brancas (Bemisia tabaci / Trialeurodes vaporariorum) em imagens
    de folhas e armadilhas, baseando-se em cor (HSV), forma (morfologia) e bordas.

Principais Funcionalidades:
    1. Pré-processamento de imagem (contraste e redução de ruído).
    2. Segmentação por cor usando faixas HSV específicas para adultos e ninfas.
    3. Filtragem de candidatos baseada em geometria (área, circularidade, solidez).
    4. Sistema de Pontuação (Scoring) para classificar a qualidade da detecção.
    5. Exportação automática para o formato YOLO (.txt).
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================

# Limiar de Confiança (0.0 a 1.0):
# Define a rigidez do filtro. Valores mais altos diminuem falsos positivos (ruído),
# mas podem perder insetos reais. 0.55 é um bom equilíbrio inicial.
DEFAULT_CONFIDENCE = 0.55

# Amostragem Visual:
# Quantidade de imagens que terão as caixas desenhadas e salvas numa pasta separada
# para que o humano possa verificar se o algoritmo está funcionando bem.
VIS_SAMPLES_PER_SPLIT = 12


class WhiteflyAutoLabelerV3:
    """
    Classe principal responsável pela orquestração do pipeline de detecção.
    Encapsula a lógica de processamento de imagem, detecção de candidatos e 
    geração de arquivos de anotação.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE, debug: bool = False):
        """
        Inicializa o detector com parâmetros de sensibilidade e define as faixas de cor.

        Args:
            confidence_threshold (float): Score mínimo para aceitar um candidato como inseto.
            debug (bool): Ativa logs detalhados para depuração.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.debug = debug

        # ======================================================================
        # DEFINIÇÃO DE FAIXAS DE COR (HSV)
        # ======================================================================
        # O espaço de cor HSV (Hue, Saturation, Value) é usado para isolar os insetos.
        # As moscas-brancas geralmente são claras/brancas/amareladas.
        
        self.stages = {
            'adulto': {
                # Faixa para moscas adultas (tons mais brancos/brilhantes)
                'hsv_lower': np.array([18, 20, 150], dtype=np.uint8),
                'hsv_upper': np.array([42, 160, 255], dtype=np.uint8),
                'min_area': 30,    # Área mínima em pixels (evita ruído granulado)
                'max_area': 3000,  # Área máxima (evita detectar a folha inteira)
                'weight': 1.0      # Peso na pontuação final
            },
            'ninfa': {
                # Faixa para ninfas (tons mais translúcidos/amarelos)
                'hsv_lower': np.array([22, 60, 120], dtype=np.uint8),
                'hsv_upper': np.array([65, 255, 255], dtype=np.uint8),
                'min_area': 10,
                'max_area': 1600,
                'weight': 0.95
            },
            'geral': {
                # Faixa abrangente para capturar casos que falham nas anteriores
                'hsv_lower': np.array([0, 0, 180], dtype=np.uint8),
                'hsv_upper': np.array([85, 255, 255], dtype=np.uint8),
                'min_area': 8,
                'max_area': 4000,
                'weight': 0.8
            }
        }

        # Referência de cor para folha (verde), usada para evitar falsos positivos
        # em áreas muito verdes.
        self.leaf_hsv_lower = np.array([30, 25, 30], dtype=np.uint8)
        self.leaf_hsv_upper = np.array([90, 255, 255], dtype=np.uint8)

        print("="*70)
        print("WHITEFLY AUTO-LABELER V3 - INICIALIZADO")
        print("="*70)
        print(f"Limiar de confiança: {self.confidence_threshold}")

    def _preprocess(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Aplica uma cadeia de filtros para preparar a imagem para segmentação.

        Pipeline:
        1. Upscaling: Aumenta imagens pequenas (<600px) para melhorar detecção de objetos pequenos.
        2. CLAHE: Equalização de histograma adaptativa no canal de Luminosidade (LAB) para contraste local.
        3. Denoising: Remove granulação (ruído) da imagem usando Non-Local Means.
        4. Blur: Suaviza bordas duras com Gaussian Blur.
        5. Canny: Detecta bordas (contornos) para auxiliar na separação de objetos.

        Args:
            img: Imagem original (BGR).

        Returns:
            Dicionário contendo a imagem em vários estágios (HSV, Gray, Edges, etc).
        """
        h, w = img.shape[:2]
        scale = 1.0

        # Redimensionamento para garantir consistência em inputs variados
        if max(h, w) < 600:
            scale = 1.5
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

        # Conversão para LAB e aplicação de CLAHE no canal de Luminosidade (L)
        # Isso destaca os insetos brancos contra o fundo verde escuro
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Redução de ruído para evitar detectar poeira como inseto
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 9, 9, 7, 21)
        blurred = cv2.GaussianBlur(denoised, (3,3), 0)

        # Geração de mapas auxiliares
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150) # Detecção de bordas

        return {
            'orig': img,
            'enhanced': enhanced,
            'hsv': hsv,
            'edges': edges,
            'scale': scale
        }

    def _stage_mask(self, hsv: np.ndarray, stage_name: str) -> np.ndarray:
        """
        Gera uma máscara binária (preto e branco) isolando pixels que correspondem
        às cores definidas em self.stages.

        Args:
            hsv: Imagem no espaço de cor HSV.
            stage_name: Chave do dicionário stages ('adulto', 'ninfa', etc).
        
        Returns:
            np.ndarray: Máscara binária processada morfologicamente.
        """
        s = self.stages[stage_name]
        # Cria máscara baseada na cor
        mask = cv2.inRange(hsv, s['hsv_lower'], s['hsv_upper'])

        # Operações Morfológicas (Opening/Closing) para limpar ruídos pequenos
        # e preencher buracos dentro dos objetos detectados.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _extract_candidates(self, pre: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Identifica contornos na imagem que podem ser moscas-brancas.
        
        Lógica:
        1. Combina a máscara de cor (HSV) com a máscara de bordas (Canny).
        2. Encontra contornos nessa máscara combinada.
        3. Filtra contornos muito pequenos ou muito grandes.
        4. Calcula métricas geométricas (circularidade, solidez, etc).

        Returns:
            Lista de dicionários, onde cada item é um candidato a detecção.
        """
        hsv = pre['hsv']
        edges = pre['edges']

        candidates = []
        
        # Itera sobre cada perfil de inseto (adulto, ninfa...)
        for stage_name in self.stages.keys():
            mask = self._stage_mask(hsv, stage_name)

            # Refina a máscara usando as bordas detectadas pelo Canny
            # Isso ajuda a separar insetos que estão muito próximos
            edges_dil = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            mask_edges = cv2.bitwise_and(mask, mask, mask=edges_dil)
            fused = cv2.bitwise_or(mask, mask_edges)

            # Encontra os contornos na máscara binária final
            cnts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sconf = self.stages[stage_name]

            for cnt in cnts:
                area = cv2.contourArea(cnt)

                # Filtro básico de tamanho
                if area < sconf['min_area'] or area > sconf['max_area']:
                    continue

                # Cálculo de propriedades geométricas para o Score
                x,y,w,h = cv2.boundingRect(cnt)
                aspect = w / (h + 1e-6) # Razão de aspecto (largura/altura)
                
                perimeter = cv2.arcLength(cnt, True)
                # Circularidade: 1.0 é um círculo perfeito. Insetos tendem a ser ovais/redondos.
                circularity = (4*np.pi*area / (perimeter*perimeter)) if perimeter > 0 else 0
                
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull) if hull is not None else 0
                # Solidez: Quão "sólido" é o contorno (sem buracos ou concavidades grandes)
                solidity = area / (hull_area + 1e-6)
                
                # Momentos de Hu: Invariantes a escala/rotação, descrevem a forma
                hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
                hu_score = 1.0/(1.0 + np.abs(hu[0]))

                candidate = {
                    'stage': stage_name,
                    'bbox': [x,y,w,h],
                    'area': area,
                    'aspect': aspect,
                    'circularity': circularity,
                    'solidity': solidity,
                    'hu_score': float(hu_score),
                    'mask_mean': float(np.mean(mask[y:y+h, x:x+w]) / 255.0),
                    'fused_mean': float(np.mean(fused[y:y+h, x:x+w]) / 255.0)
                }
                candidates.append(candidate)

        return candidates

    def _score_candidate(self, cand: Dict, pre: Dict[str, np.ndarray]) -> float:
        """
        Atribui uma nota (0.0 a 1.0) para o candidato. Quanto maior a nota,
        maior a probabilidade de ser uma mosca-branca real e não ruído.

        Fatores de pontuação:
        - Proximidade da área ideal.
        - Solidez e Circularidade (formas irregulares perdem ponto).
        - Intensidade na máscara (quão "branco" é o objeto).
        - Aspect Ratio (formas muito esticadas perdem ponto).
        
        Returns:
            float: Score normalizado entre 0.0 e 1.0.
        """
        sconf = self.stages[cand['stage']]
        score = 0.0

        # Pontuação por Área: penaliza se for muito maior ou menor que a média esperada
        ideal = (sconf['min_area'] + sconf['max_area'])/2.0
        area_score = max(0.0, 1.0 - abs(cand['area'] - ideal)/ideal)
        score += area_score * 0.30

        # Pontuação por Forma (Geometria)
        score += min(1.0, cand['solidity']) * 0.18
        score += min(1.0, cand['circularity']*2.0) * 0.15
        
        # Pontuação por preenchimento da máscara
        score += cand['mask_mean'] * 0.12
        score += cand['fused_mean'] * 0.10

        # Pontuação por Proporção (Aspect Ratio)
        # Espera-se formato oval/arredondado (~1.0 a 1.5)
        aspect_ideal = 1.2
        aspect_score = max(0.0, 1.0 - abs(cand['aspect'] - aspect_ideal)/aspect_ideal)
        score += aspect_score * 0.10

        score += min(1.0, cand['hu_score']) * 0.05

        # Aplica o peso específico do estágio (ex: 'adulto' tem peso maior que 'geral')
        score *= sconf.get('weight', 1.0)

        return float(max(0.0, min(1.0, score)))

    def _nms(self, detections: List[Dict], iou_thres: float = 0.35) -> List[Dict]:
        """
        Non-Maximum Suppression (NMS).
        
        Problema: O algoritmo pode detectar o mesmo inseto várias vezes (ex: uma caixa
        pela máscara 'adulto' e outra pela 'geral').
        
        Solução: Se duas caixas se sobrepõem significativamente (IoU > thres),
        mantém apenas a que tem o maior score e descarta a outra.
        
        Args:
            detections: Lista de candidatos pré-aprovados.
            iou_thres: Limite de sobreposição para considerar duplicata.
        """
        if not detections:
            return []

        # Ordena detecções pelo score (maior para menor)
        dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []

        for d in dets:
            x1,y1,w1,h1 = d['bbox']
            # Coordenadas da caixa atual
            box1 = [x1, y1, x1+w1, y1+h1]
            add = True

            for k in keep:
                x2,y2,w2,h2 = k['bbox']
                box2 = [x2, y2, x2+w2, y2+h2]

                # Cálculo da Interseção sobre União (IoU)
                xx1 = max(box1[0], box2[0])
                yy1 = max(box1[1], box2[1])
                xx2 = min(box1[2], box2[2])
                yy2 = min(box1[3], box2[3])

                if xx2 > xx1 and yy2 > yy1:
                    inter_area = (xx2-xx1)*(yy2-yy1)
                    box1_area = w1*h1
                    box2_area = w2*h2
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area / (union_area + 1e-6)

                    # Se sobreposição alta, descarta a caixa atual (já existe uma melhor salva)
                    if iou > iou_thres:
                        add = False
                        break

            if add:
                keep.append(d)

        return keep

    def detect_on_image(self, image_path: Path) -> List[Dict]:
        """
        Executa o pipeline completo de detecção em uma imagem.
        
        Fluxo:
        1. Carrega imagem.
        2. Pré-processa (filtros).
        3. Extrai candidatos.
        4. Calcula score de cada candidato.
        5. Remove sobreposições (NMS).
        6. Filtra pelo limiar de confiança (CONFIDENCE_THRESHOLD).
        
        Returns:
            List[Dict]: Lista final de detecções validadas.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        pre = self._preprocess(img)
        candidates = self._extract_candidates(pre)

        detections = []
        for c in candidates:
            score = self._score_candidate(c, pre)

            # Pré-filtro leve (40% do threshold) para passar para o NMS
            if score >= (self.confidence_threshold * 0.4):
                detections.append({'bbox': c['bbox'], 'score': score, 'stage': c['stage']})

        # Remove duplicatas
        detections = self._nms(detections, iou_thres=0.35)

        # Filtro final rigoroso
        final = [d for d in detections if d['score'] >= self.confidence_threshold]

        # Se a imagem foi aumentada no pré-processamento, ajusta as coordenadas de volta
        scale = pre.get('scale', 1.0)
        if scale != 1.0:
            for d in final:
                x,y,w,h = d['bbox']
                d['bbox'] = [int(x/scale), int(y/scale), int(w/scale), int(h/scale)]

        return final

    def to_yolo_lines(self, dets: List[Dict], img_w: int, img_h: int) -> List[str]:
        """
        Converte as detecções (pixels) para o formato padrão YOLO (normalizado).
        
        Formato YOLO: <classe> <centro_x> <centro_y> <largura> <altura>
        Todos os valores devem estar entre 0.0 e 1.0.
        
        Args:
            dets: Lista de detecções.
            img_w, img_h: Dimensões originais da imagem.
        """
        lines = []
        for d in dets:
            x,y,w,h = d['bbox']
            
            # Normalização
            cx = (x + w/2) / img_w
            cy = (y + h/2) / img_h
            ww = w / img_w
            hh = h / img_h

            # Clamp (garante que não passe de 1.0 ou seja menor que 0.0)
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            ww = float(np.clip(ww, 0.0, 1.0))
            hh = float(np.clip(hh, 0.0, 1.0))

            # Classe 0 (mosca-branca) fixa
            lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

        return lines

    def visualize(self, src_img_path: Path, detections: List[Dict], out_path: Path):
        """
        Gera uma imagem de debug desenhando os retângulos detectados e os scores.
        Útil para validação humana.
        
        Cores:
        - Verde: Score Alto (>=0.8)
        - Amarelo: Score Médio (>=0.6)
        - Laranja: Score Baixo
        """
        img = cv2.imread(str(src_img_path))
        if img is None:
            return

        for d in detections:
            x,y,w,h = d['bbox']
            s = d['score']

            # Cor varia conforme confiança: Verde (Alta), Amarelo (Média), Laranja (Baixa)
            color = (0,255,0) if s>=0.8 else (0,255,255) if s>=0.6 else (0,165,255)

            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{s:.2f}", (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

    def process_dataset(self, dataset_root: Path, visualize_samples: int = VIS_SAMPLES_PER_SPLIT):
        """
        Processa um dataset YOLO inteiro (pastas train/val/test).
        
        Ações:
        1. Faz backup das labels existentes (se houver) para segurança.
        2. Itera sobre todas as imagens.
        3. Gera novos arquivos .txt com as labels detectadas.
        4. Salva amostras visuais para conferência.
        5. Gera relatório JSON final.
        """
        dataset_root = Path(dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        # Backup de segurança
        labels_dir = dataset_root / 'labels'
        if labels_dir.exists():
            backup_dir = dataset_root / f"labels_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(labels_dir, backup_dir)
            if self.debug:
                print("Backup das labels criado em:", backup_dir)

        report = {
            'total_images': 0,
            'images_with_detections': 0,
            'total_detections': 0,
            'by_split': {}
        }

        vis_dir = dataset_root / 'auto_labels_v3_visual'
        vis_dir.mkdir(exist_ok=True)

        # Processa cada split separadamente
        for split in ['train','val','test']:
            imgs = list((dataset_root/'images'/split).glob('*.jpg')) + \
                   list((dataset_root/'images'/split).glob('*.png'))

            # Garante que a pasta de output existe
            (dataset_root/'labels'/split).mkdir(parents=True, exist_ok=True)

            split_stats = {'images': len(imgs), 'detections': 0, 'images_with_detections': 0}
            sample_vis = 0

            print(f"\nProcessando {split}: {len(imgs)} imagens")

            # Barra de progresso (tqdm)
            for img_path in tqdm(imgs, desc=f"{split}"):
                report['total_images'] += 1

                # Detecção
                dets = self.detect_on_image(img_path)
                
                # Leitura para obter dimensões para normalização
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h,w = img.shape[:2]
                
                # Conversão para linhas de texto YOLO
                yolo_lines = self.to_yolo_lines(dets, w, h)

                # Salva o arquivo .txt com o mesmo nome da imagem
                label_file = dataset_root/'labels'/split/f"{img_path.stem}.txt"
                label_file.write_text("\n".join(yolo_lines), encoding='utf-8')

                # Atualiza estatísticas
                if dets:
                    report['images_with_detections'] += 1
                    split_stats['images_with_detections'] += 1

                report['total_detections'] += len(dets)
                split_stats['detections'] += len(dets)

                # Gera visualização se ainda não atingiu o limite de amostras
                if sample_vis < visualize_samples and dets:
                    self.visualize(img_path, dets, vis_dir / f"{split}_{img_path.name}")
                    sample_vis += 1

            report['by_split'][split] = split_stats

        # Salva relatório técnico
        report_file = dataset_root / f'auto_label_v3_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_file.write_text(json.dumps(report, indent=2))

        print("\nAuto-labeling concluído. Relatório salvo em:", report_file)
        return report


# ==============================================================================
# PONTO DE ENTRADA (MAIN)
# ==============================================================================
if __name__ == "__main__":
    import argparse

    # Configuração de argumentos via linha de comando
    parser = argparse.ArgumentParser(description="Whitefly Auto-Labeler V3")
    parser.add_argument("--dataset", type=str, required=False,
                        default=r"C:\Users\Victor\Documents\TCC\IA\datasets\ip102_yolo_white_fly",
                        help="Caminho raiz do dataset contendo pastas images/ e labels/")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE,
                        help="Limiar de confiança das detecções (0.0 a 1.0)")
    parser.add_argument("--vis", type=int, default=VIS_SAMPLES_PER_SPLIT,
                        help="Quantidade de imagens para salvar visualização")
    parser.add_argument("--debug", action='store_true',
                        help="Ativa prints detalhados para depuração")

    args = parser.parse_args()
    root = Path(args.dataset)

    if not root.exists():
        print("ERRO: Dataset não encontrado em:", root)
        exit(1)

    # Inicializa e roda o processo
    labeler = WhiteflyAutoLabelerV3(confidence_threshold=args.conf, debug=args.debug)
    report = labeler.process_dataset(root, visualize_samples=args.vis)

    print("\nResumo final:")
    print(json.dumps(report, indent=2))
    print("\nProcesso Finalizado com Sucesso.")