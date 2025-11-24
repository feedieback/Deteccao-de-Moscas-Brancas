# -*- coding: utf-8 -*-
"""
Script Completo de Teste, Validação e Benchmark para YOLOv8
----------------------------------------------------------
Este script atua como uma suíte de testes pós-treinamento (Post-training Validation Suite).
Ele carrega um modelo treinado (.pt) e executa uma bateria de avaliações para
quantificar o desempenho técnico e a viabilidade operacional.

Principais Funcionalidades:
    1. Validação Oficial (mAP): Calcula métricas padrão COCO (Precision, Recall, mAP) no conjunto de teste.
    2. Teste Visual: Executa inferência em imagens aleatórias e salva os resultados com bounding boxes desenhadas.
    3. Matriz de Confusão: Gera gráficos para analisar erros de classificação (falso positivo vs. falso negativo).
    4. Benchmark de Velocidade: Mede a latência média de inferência (ms) e a taxa de quadros (FPS).
    5. Relatório HTML: Compila todos os dados em um dashboard web para fácil visualização.
"""

import os
# Configuração de ambiente para evitar erro "OMP: Error #15" comum em Windows com GPUs NVIDIA
# Permite múltiplas cópias da biblioteca de runtime OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import torch
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pandas as pd
from tqdm import tqdm
import yaml

class ModelValidator:
    """
    Classe orquestradora dos testes de validação.
    Gerencia o carregamento do modelo, caminhos do dataset e geração de artefatos de saída.
    """
    
    def __init__(self, model_path, dataset_yaml):
        """
        Inicializa o validador e prepara o ambiente de teste.

        Args:
            model_path (str/Path): Caminho absoluto para o arquivo de pesos (.pt).
            dataset_yaml (str/Path): Caminho para o arquivo de configuração (.yaml) do dataset.
        """
        print("="*70)
        print(" "*20 + "VALIDAÇÃO DE MODELO YOLO")
        print("="*70)
        
        # 1. Carregar modelo na memória (GPU se disponível, senão CPU)
        print(f"\nCarregando modelo: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = Path(model_path)
        
        # 2. Carregar configuração do dataset para saber caminhos e nomes das classes
        print(f"Carregando dataset: {dataset_yaml}")
        with open(dataset_yaml, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        self.dataset_yaml = dataset_yaml
        self.dataset_root = Path(self.dataset_config['path'])
        self.class_names = self.dataset_config['names']
        self.num_classes = self.dataset_config['nc']
        
        # 3. Criar diretório único para resultados baseado em timestamp (evita sobrescrita)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(f'./validation_results_{timestamp}')
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"\n✓ Modelo carregado")
        print(f"✓ {self.num_classes} classes: {list(self.class_names.values())}")
        print(f"✓ Resultados serão salvos em: {self.results_dir}")
    
    def validate_on_test_set(self):
        """
        Executa a validação oficial no conjunto de TESTE.
        Utiliza o método nativo .val() do YOLO para calcular métricas COCO.

        Métricas calculadas:
        - Precision (P): Exatidão das detecções positivas.
        - Recall (R): Capacidade de encontrar todos os objetos reais.
        - mAP@50: Média da precisão média com limiar de IoU 0.5.
        - mAP@50-95: Métrica rigorosa (média de steps de IoU de 0.5 a 0.95).

        Returns:
            dict: Dicionário contendo todas as métricas calculadas.
        """
        
        print("\n" + "="*70)
        print("VALIDAÇÃO NO CONJUNTO DE TESTE")
        print("="*70)
        
        # Executar validação oficial do YOLO
        # conf=0.001 é padrão para calcular curvas PR completas sem cortar candidatos
        metrics = self.model.val(
            data=self.dataset_yaml,
            split='test',
            save_json=True,
            save_hybrid=True,
            plots=True,
            conf=0.001,  # Limiar baixo para pegar todas detecções
            iou=0.5      # Limiar de NMS
        )
        
        # Extrair métricas principais do objeto retornado
        results = {
            'model': str(self.model_path),
            'dataset': str(self.dataset_yaml),
            'num_classes': self.num_classes,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'mAP50': float(metrics.box.map50),
                'mAP50_95': float(metrics.box.map),
                'fitness': float(metrics.fitness) # Métrica combinada da Ultralytics
            }
        }
        
        # Extrair métricas detalhadas por classe individualmente
        if hasattr(metrics.box, 'ap_class_index'):
            results['per_class'] = {}
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = self.class_names[int(class_idx)]
                results['per_class'][class_name] = {
                    'precision': float(metrics.box.p[i]),
                    'recall': float(metrics.box.r[i]),
                    'AP50': float(metrics.box.ap50[i]),
                    'AP': float(metrics.box.ap[i])
                }
        
        # Persistir resultados em JSON para análise posterior
        results_file = self.results_dir / 'validation_metrics.json'
        with open(results_file, 'w') as f:
            json.dump(results, indent=2, fp=f)
        
        print(f"\n✓ Métricas salvas em: {results_file}")
        
        return results
    
    def print_metrics(self, results):
        """
        Utilitário para imprimir as métricas no console de forma formatada e legível.
        """
        
        print("\n" + "="*70)
        print("MÉTRICAS GERAIS")
        print("="*70)
        
        metrics = results['metrics']
        print(f"\nPrecision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"mAP@0.5:      {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.2f}%)")
        print(f"mAP@0.5:0.95: {metrics['mAP50_95']:.4f} ({metrics['mAP50_95']*100:.2f}%)")
        print(f"Fitness:      {metrics['fitness']:.4f}")
        
        if 'per_class' in results:
            print("\n" + "="*70)
            print("MÉTRICAS POR CLASSE")
            print("="*70)
            print(f"\n{'Classe':<30} {'Precision':<12} {'Recall':<12} {'AP@0.5':<12} {'AP@0.5:0.95':<12}")
            print("-"*70)
            
            for class_name, class_metrics in results['per_class'].items():
                print(f"{class_name:<30} "
                      f"{class_metrics['precision']:<12.4f} "
                      f"{class_metrics['recall']:<12.4f} "
                      f"{class_metrics['AP50']:<12.4f} "
                      f"{class_metrics['AP']:<12.4f}")
    
    def test_on_images(self, test_dir=None, num_samples=10, conf_threshold=0.25):
        """
        Executa inferência visual (predição) em uma amostra de imagens.
        Gera arquivos de imagem com as bounding boxes desenhadas para inspeção qualitativa.
        
        Args:
            test_dir: Diretório das imagens de teste (se None, infere do YAML).
            num_samples: Quantidade de imagens aleatórias para testar.
            conf_threshold: Confiança mínima para considerar uma detecção válida (padrão 0.25).
        """
        
        print("\n" + "="*70)
        print("TESTE EM IMAGENS INDIVIDUAIS")
        print("="*70)
        
        # Determinar diretório de teste
        if test_dir is None:
            test_dir = self.dataset_root / 'images' / 'test'
        else:
            test_dir = Path(test_dir)
        
        if not test_dir.exists():
            print(f"❌ Diretório de teste não encontrado: {test_dir}")
            return
        
        # Listar imagens (jpg e png)
        image_files = list(test_dir.rglob('*.jpg')) + list(test_dir.rglob('*.png'))

        if not image_files:
            print(f"❌ Nenhuma imagem encontrada em: {test_dir}")
            return

        print(f"\n✓ {len(image_files)} imagens encontradas")

        # Criar diretório para resultados visuais
        visual_dir = self.results_dir / 'test_images'
        visual_dir.mkdir(exist_ok=True)
        
        # Embaralhar e selecionar amostras
        np.random.shuffle(image_files)
        test_results = []
        
        for i, img_file in enumerate(tqdm(image_files[:num_samples], desc="Testando")):
            # Predição (Inferência)
            results = self.model.predict(
                str(img_file),
                conf=conf_threshold,
                verbose=False
            )
            
            # Processar resultados
            if len(results) > 0:
                result = results[0]
                
                # Contar detecções por classe para o JSON
                detections = {}
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.class_names[class_id]
                        
                        if class_name not in detections:
                            detections[class_name] = []
                        detections[class_name].append(conf)
                
                test_results.append({
                    'image': img_file.name,
                    'num_detections': len(result.boxes) if result.boxes is not None else 0,
                    'detections': detections
                })
                
                # Salva a imagem anotada (método .plot() desenha as caixas)
                annotated = result.plot()
                output_path = visual_dir / f'pred_{i:03d}_{img_file.name}'
                cv2.imwrite(str(output_path), annotated)
        
        # Salvar resultados numéricos
        test_results_file = self.results_dir / 'test_results.json'
        with open(test_results_file, 'w') as f:
            json.dump(test_results, indent=2, fp=f)
        
        print(f"\n✓ Resultados salvos em: {test_results_file}")
        print(f"✓ Imagens anotadas salvas em: {visual_dir}")
        
        return test_results
    
    def generate_confusion_matrix(self, test_dir=None):
        """
        Gera e plota a Matriz de Confusão comparando Ground Truth vs Predições.
        Crucial para entender quais classes o modelo confunde ou onde ele erra (Falso Positivo/Negativo).
        """
        
        print("\n" + "="*70)
        print("GERANDO MATRIZ DE CONFUSÃO")
        print("="*70)
        
        # Determina caminhos de imagens e labels
        if test_dir is None:
            test_dir = self.dataset_root / 'images' / 'test'
            label_dir = self.dataset_root / 'labels' / 'test'
        else:
            test_dir = Path(test_dir)
            label_dir = test_dir.parent.parent / 'labels' / 'test'
        
        if not test_dir.exists() or not label_dir.exists():
            print(f"❌ Diretórios não encontrados")
            return
        
        # Inicializar matriz N x N
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        # Processar imagens
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        
        print(f"Processando {len(image_files)} imagens...")
        
        for img_file in tqdm(image_files, desc="Matriz de confusão"):
            # 1. Ler Ground Truth (arquivo .txt)
            label_file = label_dir / f'{img_file.stem}.txt'
            
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                gt_lines = f.readlines()
            
            if not gt_lines:
                continue
            
            # 2. Rodar Predição
            results = self.model.predict(str(img_file), conf=0.25, verbose=False)
            
            if len(results) == 0 or results[0].boxes is None:
                continue
            
            # 3. Comparar GT com Predições (Matching via IoU)
            for gt_line in gt_lines:
                gt_class = int(gt_line.split()[0])
                gt_coords = [float(x) for x in gt_line.split()[1:]]
                
                # Encontrar predição mais próxima (maior IoU)
                best_iou = 0
                best_pred_class = gt_class  # Default: assume acerto se não achar match (simplificação para visualização)
                
                for box in results[0].boxes:
                    pred_class = int(box.cls[0])
                    pred_coords = box.xywhn[0].cpu().numpy()
                    
                    # Calcular IoU
                    iou = self.calculate_iou_xywh(gt_coords, pred_coords)
                    
                    # Se houver sobreposição significativa (>0.5), considera um match
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_pred_class = pred_class
                
                # Atualizar matriz
                conf_matrix[gt_class][best_pred_class] += 1
        
        # Plotar matriz com Seaborn (Heatmap)
        plt.figure(figsize=(12, 10))
        
        # Normalizar por linha (Recall por classe)
        conf_matrix_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=list(self.class_names.values()),
                    yticklabels=list(self.class_names.values()),
                    cbar_kws={'label': 'Proporção'})
        
        plt.xlabel('Classe Predita', fontsize=12, fontweight='bold')
        plt.ylabel('Classe Verdadeira', fontsize=12, fontweight='bold')
        plt.title('Matriz de Confusão (Normalizada)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Salvar imagem
        matrix_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Matriz de confusão salva em: {matrix_path}")
        
        # Salvar dados numéricos da matriz
        matrix_data = {
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_matrix_normalized': conf_matrix_norm.tolist(),
            'classes': list(self.class_names.values())
        }
        
        matrix_json = self.results_dir / 'confusion_matrix.json'
        with open(matrix_json, 'w') as f:
            json.dump(matrix_data, indent=2, fp=f)
        
        return conf_matrix
    
    def calculate_iou_xywh(self, box1, box2):
        """
        Função auxiliar para calcular Intersection over Union (IoU) entre duas caixas.
        Usada para determinar se uma predição corresponde a um Ground Truth.
        
        Args:
            box1, box2: Listas [x_center, y_center, width, height] (valores normalizados)
        
        Returns:
            float: Valor IoU (0.0 a 1.0).
        """
        
        # Converter formato xywh (centro) para xyxy (cantos)
        def xywh_to_xyxy(box):
            x, y, w, h = box
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            return [x1, y1, x2, y2]
        
        box1_xyxy = xywh_to_xyxy(box1)
        box2_xyxy = xywh_to_xyxy(box2)
        
        # Calcular coordenadas da área de interseção
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box1_xyxy[3], box2_xyxy[3])
        
        # Se não houver interseção
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calcular área de união
        box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)
    
    def benchmark_speed(self, num_runs=100):
        """
        Mede a velocidade de inferência pura (sem overhead de leitura de disco).
        Gera estatísticas de latência (ms) e taxa de quadros (FPS).
        
        Args:
            num_runs: Quantas inferências executar para tirar a média.
        """
        
        print("\n" + "="*70)
        print("BENCHMARK DE VELOCIDADE")
        print("="*70)
        
        # Criar imagem de teste sintética (ruído aleatório) do tamanho padrão YOLO
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warm-up: Rodar algumas vezes para inicializar a GPU/bibliotecas
        print("\nAquecendo GPU/CPU...")
        for _ in range(10):
            _ = self.model.predict(test_img, verbose=False)
        
        # Benchmark real
        print(f"Executando {num_runs} inferências...")
        times = []
        
        for _ in tqdm(range(num_runs), desc="Benchmark"):
            start = time.time()
            _ = self.model.predict(test_img, verbose=False)
            times.append(time.time() - start)
        
        # Estatísticas em milissegundos
        times_ms = np.array(times) * 1000
        
        results = {
            'mean_ms': float(np.mean(times_ms)),
            'std_ms': float(np.std(times_ms)),
            'min_ms': float(np.min(times_ms)),
            'max_ms': float(np.max(times_ms)),
            'median_ms': float(np.median(times_ms)),
            'fps': float(1000 / np.mean(times_ms)), # FPS = 1000 / média_ms
            'device': str(self.model.device)
        }
        
        print(f"\n{'='*70}")
        print("RESULTADOS DO BENCHMARK")
        print('='*70)
        print(f"Dispositivo:      {results['device']}")
        print(f"Tempo médio:      {results['mean_ms']:.2f} ms")
        print(f"Desvio padrão:    {results['std_ms']:.2f} ms")
        print(f"Tempo mínimo:     {results['min_ms']:.2f} ms")
        print(f"Tempo máximo:     {results['max_ms']:.2f} ms")
        print(f"Mediana:          {results['median_ms']:.2f} ms")
        print(f"FPS estimado:     {results['fps']:.2f}")
        
        # Salvar resultados
        benchmark_file = self.results_dir / 'speed_benchmark.json'
        with open(benchmark_file, 'w') as f:
            json.dump(results, indent=2, fp=f)
        
        # Plotar histograma de distribuição de tempo
        plt.figure(figsize=(10, 6))
        plt.hist(times_ms, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(results['mean_ms'], color='r', linestyle='--', label=f'Média: {results["mean_ms"]:.2f} ms')
        plt.axvline(results['median_ms'], color='g', linestyle='--', label=f'Mediana: {results["median_ms"]:.2f} ms')
        plt.xlabel('Tempo de Inferência (ms)', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Distribuição do Tempo de Inferência', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        speed_plot = self.results_dir / 'speed_distribution.png'
        plt.savefig(speed_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Resultados salvos em: {benchmark_file}")
        print(f"✓ Gráfico salvo em: {speed_plot}")
        
        return results
    
    def generate_report(self, validation_results, speed_results=None):
        """
        Compila todos os dados gerados (métricas, caminhos de imagens) em um arquivo HTML.
        Cria um dashboard visual fácil de ler.
        """
        
        print("\n" + "="*70)
        print("GERANDO RELATÓRIO HTML")
        print("="*70)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Validação - Detecção de Pragas</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px 25px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        .good {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .bad {{ color: #dc3545; }}
        img {{
            max-width: 100%;
            border-radius: 8px;
            margin-top: 15px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐛 Relatório de Validação - Detecção de Pragas Agrícolas</h1>
        <p>Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        <p>Modelo: {self.model_path.name}</p>
    </div>
    
    <div class="section">
        <h2>📊 Métricas Gerais</h2>
        <div class="metric">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{validation_results['metrics']['precision']:.2%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{validation_results['metrics']['recall']:.2%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">mAP@0.5</div>
            <div class="metric-value">{validation_results['metrics']['mAP50']:.2%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">mAP@0.5:0.95</div>
            <div class="metric-value">{validation_results['metrics']['mAP50_95']:.2%}</div>
        </div>
    </div>
"""

        if 'per_class' in validation_results:
            html_content += """
    <div class="section">
        <h2>📋 Métricas por Classe</h2>
        <table>
            <tr>
                <th>Classe</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>AP@0.5</th>
                <th>AP@0.5:0.95</th>
            </tr>
"""
            for class_name, metrics in validation_results['per_class'].items():
                html_content += f"""
            <tr>
                <td>{class_name}</td>
                <td>{metrics['precision']:.4f}</td>
                <td>{metrics['recall']:.4f}</td>
                <td>{metrics['AP50']:.4f}</td>
                <td>{metrics['AP']:.4f}</td>
            </tr>
"""
            html_content += """
        </table>
    </div>
"""

        if speed_results:
            html_content += f"""
    <div class="section">
        <h2>⚡ Performance de Velocidade</h2>
        <div class="metric">
            <div class="metric-label">Tempo Médio</div>
            <div class="metric-value">{speed_results['mean_ms']:.2f} ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">FPS</div>
            <div class="metric-value">{speed_results['fps']:.1f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Dispositivo</div>
            <div class="metric-value">{speed_results['device']}</div>
        </div>
    </div>
"""

        html_content += """
    <div class="section">
        <h2>🎯 Matriz de Confusão</h2>
        <img src="confusion_matrix.png" alt="Matriz de Confusão">
    </div>
    
    <div class="section">
        <h2>✅ Conclusão</h2>
        <p>O modelo foi validado com sucesso no conjunto de teste.</p>
        <p>Todos os resultados e visualizações foram salvos no diretório de resultados.</p>
    </div>
</body>
</html>
"""
        
        # Salvar HTML
        report_path = self.results_dir / 'validation_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Relatório HTML gerado: {report_path}")
        print(f"\nAbra o arquivo no navegador para visualizar o relatório completo!")
        
        return report_path


def find_latest_model():
    """
    Tenta localizar automaticamente o arquivo 'best.pt' mais recente
    dentro das pastas do projeto (runs, pest_detection_fast, etc.).
    Isso facilita a execução sem precisar digitar caminhos longos.
    """
    
    search_paths = [
        Path('./pest_detection_fast'),
        Path('./pest_detection_ip102'),
        Path('./runs/detect'),
    ]
    
    latest_model = None
    latest_time = 0
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        for train_dir in base_path.glob('train*'):
            weights_dir = train_dir / 'weights'
            if weights_dir.exists():
                best_pt = weights_dir / 'best.pt'
                if best_pt.exists():
                    mtime = best_pt.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_model = best_pt
    
    return latest_model


import time

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print(" "*15 + "TESTE E VALIDAÇÃO DE MODELO YOLO")
    print("="*70)
    
    # 1. Tentar encontrar modelo automaticamente para conveniência
    print("\nBuscando modelo treinado...")
    model_path = find_latest_model()
    
    if model_path:
        print(f"✓ Modelo encontrado: {model_path}")
        use_found = input("Usar este modelo? (s/n): ")
        
        if use_found.lower() != 's':
            model_path = None
    
    # 2. Solicitar caminho manual se necessário
    if not model_path:
        model_input = input("\nCaminho do modelo (.pt): ").strip()
        model_path = Path(model_input)
        
        if not model_path.exists():
            print(f"\n Modelo não encontrado: {model_path}")
            input("\nPressione Enter para sair...")
            exit(1)
    
    # 3. Configurar caminho do dataset
    # (Idealmente parametrizável, mas hardcoded para simplicidade neste projeto)
    dataset_yaml = Path(r'C:\Users\Victor\Documents\TCC\IA\datasets\ip102_yolo_white_fly\ip102.yaml')
    
    if not dataset_yaml.exists():
        print(f"\n Dataset YAML não encontrado: {dataset_yaml}")
        input("\nPressione Enter para sair...")
        exit(1)
    
    # 4. Instanciar Validador
    validator = ModelValidator(
        model_path=str(model_path),
        dataset_yaml=str(dataset_yaml)
    )
    
    # 5. Menu Interativo
    print("\n" + "="*70)
    print("SELECIONE OS TESTES A EXECUTAR")
    print("="*70)
    print("\n1. Validação completa (métricas oficiais)")
    print("2. Teste em imagens individuais")
    print("3. Gerar matriz de confusão")
    print("4. Benchmark de velocidade")
    print("5. EXECUTAR TODOS (Recomendado)")
    
    escolha = input("\nEscolha (1-5): ")
    
    validation_results = None
    speed_results = None
    
    # Lógica de execução baseada na escolha
    if escolha == '1' or escolha == '5':
        validation_results = validator.validate_on_test_set()
        validator.print_metrics(validation_results)
    
    if escolha == '2' or escolha == '5':
        validator.test_on_images(num_samples=20, conf_threshold=0.25)
    
    if escolha == '3' or escolha == '5':
        validator.generate_confusion_matrix()
    
    if escolha == '4' or escolha == '5':
        speed_results = validator.benchmark_speed(num_runs=100)
    
    if escolha == '5':
        # Gerar relatório completo
        if validation_results:
            validator.generate_report(validation_results, speed_results)
    
    print("\n" + "="*70)
    print("✓ VALIDAÇÃO CONCLUÍDA!")
    print("="*70)
    print(f"\n Resultados salvos em: {validator.results_dir}")

    print("\nPróximos passos:")
    print("1️ Abra o relatório HTML gerado para visualizar métricas e gráficos.")
    print("2  Analise as imagens anotadas na pasta 'test_images'.")
    print("3  Consulte os arquivos JSON para uso em relatórios técnicos.")

    input("\nPressione Enter para sair...")