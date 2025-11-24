# -*- coding: utf-8 -*-
"""
Treinador YOLO Otimizado para Mosca-Branca (Trialeurodes vaporariorum)
----------------------------------------------------------------------
Script de automação para treinamento de modelos YOLOv8 com foco em:
1. Detecção de objetos pequenos (Small Object Detection).
2. Estabilidade em GPUs com pouca VRAM (ex: GTX 1650 4GB).
3. Curadoria automática de dados (conversão de classes e split de validação).

Funcionalidades Principais:
    - Pipeline de Dados: Verifica integridade, cria backup e converte labels
      multiclasse para classe única (0 = whitefly).
    - Gestão de Recursos: Monitora erros de memória (CUDA OOM) e reduz
      automaticamente o `batch_size` para tentar recuperar o treino.
    - Hiperparâmetros Customizados: Configuração específica para evitar
      a distorção de insetos pequenos (Mosaic=0.0) e garantir convergência (AdamW).
"""

import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime
import random

# Configurações de ambiente para evitar conflitos comuns em Windows/Intel
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from ultralytics import YOLO
import torch
import yaml

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS E CONSTANTES
# ==============================================================================

# Caminho absoluto para o arquivo de configuração do dataset
# IMPORTANTE: Este arquivo define onde estão as imagens e nomes das classes
DATASET_YAML_PATH = Path(r"C:\Users\Victor\Documents\TCC\IA\datasets\ip102_yolo_white_fly\ip102.yaml")

# Configuração de Mapeamento de Classes
# O dataset IP102 original tem 102 classes. A mosca-branca é a classe 5.
# Este script irá filtrar apenas a classe 5 e reescrevê-la como 0 (classe única).
TARGET_ORIG_CLASS = 5   

# Diretório de segurança para salvar as labels originais antes de modificar
BACKUP_DIR = Path(r"C:\Users\Victor\Documents\TCC\IA\datasets\ip102_yolo_white_fly\labels")

# Configurações Padrão do Modelo
DEFAULT_MODEL = "yolov8s"   # 's' (small) é melhor que 'n' (nano) para detalhes finos
DEFAULT_PROJECT = "whitefly_detection_opt"
MAX_RETRIES_ON_OOM = 2      # Quantas vezes tentar reduzir o batch se a memória acabar


def read_yaml(path: Path):
    """Lê um arquivo YAML e retorna como dicionário Python."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data):
    """Salva um dicionário Python como arquivo YAML."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def ensure_label_backup(dataset_root: Path):
    """
    Mecanismo de Segurança:
    Cria uma cópia completa da pasta 'labels' antes de qualquer alteração.
    Isso permite restaurar o dataset original caso a conversão falhe.
    """
    if BACKUP_DIR.exists():
        print(f"Backup já existe em: {BACKUP_DIR}")
        return
    print(f"Criando backup das labels em: {BACKUP_DIR} ...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    orig_labels = dataset_root / "labels"
    if not orig_labels.exists():
        print("Pasta de labels original não encontrada; nada a copiar.")
        return
    shutil.copytree(orig_labels, BACKUP_DIR / "labels", dirs_exist_ok=True)
    print("Backup criado.")


def convert_labels_to_single_class(dataset_root: Path, target_orig_class: int = TARGET_ORIG_CLASS):
    """
    Processador de Labels:
    Lê todos os arquivos .txt do dataset.
    1. Filtra: Mantém apenas linhas onde a classe é TARGET_ORIG_CLASS (5).
    2. Converte: Muda o ID da classe para 0 (padrão YOLO para single-class).
    3. Limpa: Se a imagem não tiver a classe alvo, o arquivo fica vazio (imagem negativa).
    
    Args:
        dataset_root: Caminho raiz do dataset.
        target_orig_class: ID da classe original a ser preservada.
    """

    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    if not labels_root.exists():
        print("Pasta de labels não encontrada:", labels_root)
        return

    print("Convertendo labels para classe única (0 = whitefly).")
    ensure_label_backup(dataset_root)

    kept = 0
    removed = 0
    for split in ("train", "val", "test"):
        lbl_dir = labels_root / split
        img_dir = images_root / split
        if not lbl_dir.exists():
            continue
        for txt_path in list(lbl_dir.rglob("*.txt")):
            try:
                lines = txt_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                lines = []
            new_lines = []
            for L in lines:
                parts = L.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                except:
                    continue
                # Lógica de Filtro: Manter apenas se for a classe alvo (5)
                if cls == target_orig_class:
                    # Reescreve mudando a classe para 0
                    new_lines.append("0 " + " ".join(parts[1:]))
            if new_lines:
                txt_path.write_text("\n".join(new_lines), encoding="utf-8")
                kept += 1
                # Verifica integridade: imagem existe para este label?
                img_jpg = img_dir / (txt_path.stem + ".jpg")
                img_png = img_dir / (txt_path.stem + ".png")
                if not (img_jpg.exists() or img_png.exists()):
                    print("Atenção: label existe mas imagem não encontrada:", txt_path)
            else:
                # Mantemos o arquivo vazio para o YOLO saber que é uma imagem de fundo (sem objeto)
                txt_path.write_text("", encoding="utf-8")
                removed += 1


    print(f"Conversão concluída. Mantidos: {kept} labels; removidos: {removed} labels.")


def verify_and_prepare_dataset(yaml_path: Path):
    """
    Orquestrador de Preparação de Dados:
    1. Lê o YAML para entender a estrutura de pastas.
    2. Verifica a existência de labels de validação.
    3. AUTO-SPLIT: Se não houver validação, move aleatoriamente 10% do treino para validação.
    4. Chama a conversão de classes se detectar IDs diferentes de 0.
    
    Returns:
        Path: Caminho raiz do dataset preparado.
    """

    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {yaml_path}")

    cfg = read_yaml(yaml_path)
    root = Path(cfg.get("path", ".")).resolve()

    # Resolução de caminhos (suporta caminhos relativos ou absolutos)
    train_images = root / cfg.get("train", "images/train")
    val_images = root / cfg.get("val", "images/val")
    test_images = root / cfg.get("test", "images/test")

    train_labels = root / "labels/train"    
    val_labels   = root / "labels/val"        
    test_labels  = root / "labels/test"       

    # Verificações iniciais
    val_has_labels = val_labels.exists() and any(val_labels.rglob("*.txt"))
    train_has_labels = train_labels.exists() and any(train_labels.rglob("*.txt"))

    print("Dataset root:", root)
    # ... (prints de debug omitidos) ...

    # Detecção Automática de Classes
    # Lê uma amostra para saber se precisamos converter (ex: se achar classe 5)
    classes_found = set()
    if train_labels.exists():
        for i, txt in enumerate(train_labels.rglob("*.txt")):
            if i >= 200:  # Limite de amostra para performance
                break
            try:
                for L in txt.read_text(encoding="utf-8").splitlines():
                    parts = L.strip().split()
                    if parts:
                        classes_found.add(int(parts[0]))
            except Exception:
                continue

    print("Classes encontradas nas labels (amostra):", classes_found)

    # Se encontrar classes "estranhas" (não 0), converte tudo
    if classes_found and (classes_found != {0}):
        convert_labels_to_single_class(root, TARGET_ORIG_CLASS)

    # Re-verificação após conversão
    val_has_labels = val_labels.exists() and any(val_labels.rglob("*.txt"))
    train_has_labels = train_labels.exists() and any(train_labels.rglob("*.txt"))
    
    # Criação Automática de Validação (Split)
    # Se tiver treino mas não tiver validação, cria o conjunto de validação movendo arquivos.
    if (not val_has_labels) and train_has_labels:
        print("Val set sem labels detectados — criando split automático (10% do train) para val).")
        
        train_label_files = list(train_labels.rglob("*.txt"))
        random.seed(42) # Seed fixa para reprodutibilidade
        random.shuffle(train_label_files)
        
        n_val = max(1, int(0.10 * len(train_label_files))) # 10% para validação
        val_selection = train_label_files[:n_val]
        
        for txt in val_selection:
            # Move label (.txt)
            rel = txt.relative_to(train_labels)
            target_txt = val_labels / rel
            target_txt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(txt), str(target_txt))

            # Move imagem correspondente (.jpg ou .png)
            img_jpg = (train_images / rel.with_suffix(".jpg").name)
            img_png = (train_images / rel.with_suffix(".png").name)
            if img_jpg.exists():
                (val_images / img_jpg.name).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_jpg), str(val_images / img_jpg.name))
            elif img_png.exists():
                (val_images / img_png.name).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_png), str(val_images / img_png.name))

        print(f"Movidos {n_val} amostras do train -> val para validação.")

    # Verificação Final
    n_train_imgs = sum(1 for _ in (train_images.rglob("*.jpg")))
    n_train_labels = sum(1 for _ in (train_labels.rglob("*.txt")))
    print(f"Train images: {n_train_imgs:,}, train labels: {n_train_labels:,}")
    if n_train_labels == 0:
        raise RuntimeError("Nenhuma label encontrada no train após preparação — verifique o dataset.")

    return root


def try_train_with_retries(model: YOLO, train_args: dict, max_retries=MAX_RETRIES_ON_OOM):
    """
    Wrapper de Robustez para Treinamento:
    Tenta executar model.train(). Se encontrar erro de "Out of Memory" (CUDA OOM),
    reduz o tamanho do lote (batch size) pela metade e tenta novamente.
    
    Args:
        model: Instância YOLO carregada.
        train_args: Dicionário de hiperparâmetros.
        max_retries: Número máximo de tentativas de redução.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            torch.cuda.empty_cache() # Limpa VRAM antes de tentar
            print(f"\nIniciando tentativa de treino (tentativa {attempt+1})...")
            results = model.train(**train_args)
            return results
        except RuntimeError as e:
            msg = str(e).lower()
            # Detecta erro de memória da GPU
            if "out of memory" in msg or "cuda error" in msg:
                attempt += 1
                if "batch" in train_args and train_args["batch"] > 1:
                    old = train_args["batch"]
                    # Reduz batch pela metade (ex: 16 -> 8 -> 4)
                    train_args["batch"] = max(1, int(old // 2))
                    print(f"OOM detectado: reduzindo batch de {old} -> {train_args['batch']} e reiniciando...")
                    torch.cuda.empty_cache()
                    time.sleep(3)
                    continue
                else:
                    raise # Não dá para reduzir mais, lança o erro
            else:
                raise # Outro tipo de erro, lança normalmente


def build_train_args(yaml_path: Path, model_version=DEFAULT_MODEL, project=DEFAULT_PROJECT):
    """
    Construtor de Hiperparâmetros Otimizados:
    Define a estratégia de treinamento focada em PEQUENOS OBJETOS e HARDWARE LIMITADO.
    
    Principais Ajustes:
    - imgsz=512: Equilíbrio entre detalhe visual e uso de memória.
    - optimizer='AdamW': Convergência mais estável que SGD.
    - mosaic=0.0: DESATIVADO. Mosaic mistura 4 imagens e diminui o tamanho relativo
      dos objetos. Como a mosca-branca já é minúscula, o mosaic prejudica o aprendizado.
    - amp=False: Desativado para evitar instabilidade (NaN) em placas GTX série 16xx.
    """
    args = dict(
        data=str(yaml_path),

        # --- Ajustes de Hardware (GTX 1650 / 4GB VRAM) ---
        imgsz=512,           # Resolução de entrada
        batch=13,            # Batch size inicial (será reduzido se der OOM)
        workers=1,           # Workers baixos para evitar overhead de CPU no Windows

        epochs=200,
        cache="disk",        # Cache em disco para economizar RAM
        patience=60,         # Early Stopping: para se não melhorar em 60 épocas
        device=0 if torch.cuda.is_available() else "cpu",
        project=project,
        name=f"whitefly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

        # --- Estratégia de Otimização ---
        optimizer="AdamW",   # Otimizador moderno
        lr0=0.001,           # Taxa de aprendizado inicial
        amp=False,           # Precision plena (FP32) para estabilidade numérica
        half=False,
        single_cls=True,     # Força o modelo a tratar tudo como uma única classe

        # --- Data Augmentation (Específico para Insetos) ---
        # Augmentações "destrutivas" desligadas para preservar morfologia
        multi_scale=False,   
        mosaic=0.0,          # CRÍTICO: Desligado para não reduzir o inseto
        mixup=0.0,
        auto_augment=None,

        # Augmentações Geométricas Leves (Preservam o objeto)
        degrees=10.0,        # Rotação leve
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,          # Espelhamento horizontal
        erasing=0.1,         # Random Erasing leve para robustez a oclusão

        # --- Ajustes de Cor (HSV) ---
        # Suavizados para não alterar a cor branca característica da praga
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,

        # --- Outros ---
        overlap_mask=True,
        save_period=-1,      # Salva apenas o melhor e o último checkpoint
        verbose=True,
        plots=False
    )
    
    # Define o peso inicial do modelo
    args["model"] = f"{model_version}.pt"
    return args

# (Nota: Esta função parece ser uma duplicata da definida acima, mantida conforme o código original)
def convert_labels_to_single_class(dataset_root: Path, target_orig_class: int = TARGET_ORIG_CLASS):
    """
    Versão secundária da função de conversão de labels (mesma lógica da anterior).
    Garante que apenas a classe alvo seja mantida e convertida para ID 0.
    """
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    print("Convertendo labels para classe única (0 = whitefly).")
    ensure_label_backup(dataset_root)

    kept = 0
    removed = 0

    for split in ("train", "val", "test"):
        lbl_dir = labels_root / split
        img_dir = images_root / split

        if not lbl_dir.exists():
            continue

        for txt_path in list(lbl_dir.rglob("*.txt")):
            try:
                lines = txt_path.read_text(encoding="utf-8").splitlines()
            except:
                lines = []

            new_lines = []
            for L in lines:
                parts = L.strip().split()
                if len(parts) >= 5:
                    try:
                        cls = int(parts[0])
                    except:
                        continue

                    if cls == target_orig_class:
                        new_lines.append("0 " + " ".join(parts[1:]))

            if new_lines:
                txt_path.write_text("\n".join(new_lines), encoding="utf-8")
                kept += 1
            else:
                # Importante: Não deletar o arquivo, apenas limpar o conteúdo.
                # Isso indica ao YOLO que a imagem é um "negative sample" (apenas fundo).
                txt_path.write_text("", encoding="utf-8")
                removed += 1

    print(f"Conversão concluída. Mantidos: {kept}, esvaziados: {removed}.")


def main():
    """
    Função Principal (Entry Point):
    1. Verifica e prepara o dataset (splits, classes).
    2. Carrega o modelo pré-treinado (Transfer Learning).
    3. Constrói os argumentos de treino otimizados.
    4. Executa o loop de treinamento com tolerância a falhas (OOM).
    5. Executa validação final.
    """
    print("\n=== Iniciando preparação e treino (whitefly) ===\n")

    # Validação do arquivo de configuração
    yaml_path = DATASET_YAML_PATH
    if not yaml_path.exists():
        print("Erro: YAML não encontrado:", yaml_path)
        sys.exit(1)

    # Etapa de Preparação de Dados
    dataset_root = verify_and_prepare_dataset(yaml_path)
    print("\nDataset preparado em:", dataset_root)

    # Carregamento do Modelo Base (COCO Pre-trained)
    model_version = DEFAULT_MODEL
    model = YOLO(f"{model_version}.pt")
    print("Modelo carregado:", model_version)

    # Configuração dos Hiperparâmetros
    train_args = build_train_args(yaml_path, model_version=model_version, project=DEFAULT_PROJECT)

    # Ajuste técnico para API da Ultralytics
    if "model" in train_args:
        train_args.pop("model")

    # Limpeza de memória antes do início
    torch.cuda.empty_cache()

    try:
        # Início do Treinamento (com retry automático)
        results = try_train_with_retries(model, train_args, max_retries=MAX_RETRIES_ON_OOM)
    except Exception as e:
        print("Erro durante o treinamento:", e)
        raise

    print("\nTreinamento finalizado. Resultado salvo em:", results)
    
    # Validação Final Automática
    try:
        print("\nExecutando validação final (self.model.val)...")
        model.val(plots=True)
    except Exception as e:
        print("Validação final falhou:", e)

    torch.cuda.empty_cache()
    print("\n=== Processo concluído ===\n")


if __name__ == "__main__":
    main()