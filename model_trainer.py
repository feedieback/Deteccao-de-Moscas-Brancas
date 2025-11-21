# -*- coding: utf-8 -*-
"""
Treinador YOLO otimizado para detecção de MOSCA-BRANCA (whitefly)

Características:
- Mantém o caminho do YAML conforme fornecido (não modificado)
- Verifica/convete labels para 1 classe (0 = whitefly) automaticamente com backup
- Cria val set (split) se estiver ausente
- Parâmetros otimizados para objetos pequenos (imgsz=640, mosaic elevado, FP16)
- Tenta recuperar de OOM reduzindo batch automaticamente
"""

import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime
import random

# evitar erro OpenMP múltiplo no Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from ultralytics import YOLO
import torch
import yaml

# -------------- CONFIGURÁVEIS --------------
# Mantenha este caminho exatamente como seu YAML correto
DATASET_YAML_PATH = Path(r"C:\Users\Victor\Documents\TCC\IA\datasets\ip102_yolo_white_fly\ip102.yaml")

# Índice original da mosca-branca no seu dataset IP102 (mudar se necessário)
TARGET_ORIG_CLASS = 5   # geralmente 5 = Trialeurodes_vaporariorum no IP102

# Backup das labels originais será criado aqui
BACKUP_DIR = Path(r"C:\Users\Victor\Documents\TCC\IA\datasets\ip102_yolo_white_fly\labels")

# Hyperparams
DEFAULT_MODEL = "yolov8s"   # sugerido para objetos pequenos (pode usar 'yolov8n' se preferir)
DEFAULT_PROJECT = "whitefly_detection_opt"
MAX_RETRIES_ON_OOM = 2
# -------------------------------------------


def read_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def ensure_label_backup(dataset_root: Path):
    """Cria backup (apenas uma vez) das labels originais antes de qualquer modificação."""
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
    Percorre labels em train/val/test e:
    - Mantém apenas as linhas cuja classe == target_orig_class
    - Reescreve essas linhas trocando a classe para 0
    - Remove arquivos de label vazios resultantes
    - Copia apenas imagens que tiveram labels mantidos
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
                # manter apenas se for a classe alvo original
                if cls == target_orig_class:
                    new_lines.append("0 " + " ".join(parts[1:]))
            if new_lines:
                txt_path.write_text("\n".join(new_lines), encoding="utf-8")
                kept += 1
                # garantir que a imagem correspondente exista; se sim, manter/confirmar
                img_jpg = img_dir / (txt_path.stem + ".jpg")
                img_png = img_dir / (txt_path.stem + ".png")
                if not (img_jpg.exists() or img_png.exists()):
                    print("Atenção: label existe mas imagem não encontrada:", txt_path)
            else:
                # NÃO apagar labels! Apenas ignorar se quiser.
                # Mantemos o arquivo vazio, pois o YOLO aceita normalmente.
                txt_path.write_text("", encoding="utf-8")
                removed += 1


    print(f"Conversão concluída. Mantidos: {kept} labels; removidos: {removed} labels.")


def verify_and_prepare_dataset(yaml_path: Path):
    """
    Verifica estrutura do dataset a partir do YAML, cria val split se necessário,
    converte labels para single-class se necessárias.
    Retorna path raiz do dataset (campo 'path' do YAML).
    """

    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {yaml_path}")

    cfg = read_yaml(yaml_path)
    root = Path(cfg.get("path", ".")).resolve()

    # normalizar strings (pode ser relativo ao yaml 'path')
    train_images = root / cfg.get("train", "images/train")
    val_images = root / cfg.get("val", "images/val")
    test_images = root / cfg.get("test", "images/test")

    train_labels = root / "labels/train"    
    val_labels   = root / "labels/val"        
    test_labels  = root / "labels/test"       


    # se val_labels não existir ou estiver vazio, faremos um split automático
    val_has_labels = val_labels.exists() and any(val_labels.rglob("*.txt"))
    train_has_labels = train_labels.exists() and any(train_labels.rglob("*.txt"))

    print("Dataset root:", root)
    print("Train images:", train_images)
    print("Val images:", val_images)
    print("Train labels:", train_labels)
    print("Val labels:", val_labels)

    # Convert labels to single class if necessary (detect if there are non-zero classes)
    # Check a sample of labels to detect classes present
    classes_found = set()
    if train_labels.exists():
        for i, txt in enumerate(train_labels.rglob("*.txt")):
            if i >= 200:  # sample limit
                break
            try:
                for L in txt.read_text(encoding="utf-8").splitlines():
                    parts = L.strip().split()
                    if parts:
                        classes_found.add(int(parts[0]))
            except Exception:
                continue

    print("Classes encontradas nas labels (amostra):", classes_found)

    if classes_found and (classes_found != {0}):
        # converter automaticamente mantendo backup
        convert_labels_to_single_class(root, TARGET_ORIG_CLASS)

    # Re-check validation labels
    val_has_labels = val_labels.exists() and any(val_labels.rglob("*.txt"))
    train_has_labels = train_labels.exists() and any(train_labels.rglob("*.txt"))
    # If val has no labels but train has, create val split from train (10%)
    if (not val_has_labels) and train_has_labels:
        print("Val set sem labels detectados — criando split automático (10% do train) para val).")
        # gather all train images that have labels
        train_label_files = list(train_labels.rglob("*.txt"))
        random.seed(42)
        random.shuffle(train_label_files)
        n_val = max(1, int(0.10 * len(train_label_files)))
        val_selection = train_label_files[:n_val]
        for txt in val_selection:
            # move txt and corresponding image to val folders
            rel = txt.relative_to(train_labels)
            target_txt = val_labels / rel
            target_txt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(txt), str(target_txt))

            # move image
            img_jpg = (train_images / rel.with_suffix(".jpg").name)
            img_png = (train_images / rel.with_suffix(".png").name)
            if img_jpg.exists():
                (val_images / img_jpg.name).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_jpg), str(val_images / img_jpg.name))
            elif img_png.exists():
                (val_images / img_png.name).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_png), str(val_images / img_png.name))

        print(f"Movidos {n_val} amostras do train -> val para validação.")

    # final quick checks
    n_train_imgs = sum(1 for _ in (train_images.rglob("*.jpg")))
    n_train_labels = sum(1 for _ in (train_labels.rglob("*.txt")))
    print(f"Train images: {n_train_imgs:,}, train labels: {n_train_labels:,}")
    if n_train_labels == 0:
        raise RuntimeError("Nenhuma label encontrada no train após preparação — verifique o dataset.")

    return root


def try_train_with_retries(model: YOLO, train_args: dict, max_retries=MAX_RETRIES_ON_OOM):
    """Tenta treinar e, em caso de OOM CUDA, reduz batch e tenta novamente."""
    attempt = 0
    while attempt <= max_retries:
        try:
            torch.cuda.empty_cache()
            print(f"\nIniciando tentativa de treino (tentativa {attempt+1})...")
            results = model.train(**train_args)
            return results
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                attempt += 1
                # reduzir batch para tentar recuperar
                if "batch" in train_args and train_args["batch"] > 1:
                    old = train_args["batch"]
                    train_args["batch"] = max(1, int(old // 2))
                    print(f"OOM detectado: reduzindo batch de {old} -> {train_args['batch']} e reiniciando...")
                    torch.cuda.empty_cache()
                    time.sleep(3)
                    continue
                else:
                    raise
            else:
                # re-raise for other exceptions
                raise


def build_train_args(yaml_path: Path, model_version=DEFAULT_MODEL, project=DEFAULT_PROJECT):
    """Constrói dicionário de argumentos para model.train() com boas práticas para whitefly"""
    args = dict(
        data=str(yaml_path),

        # --- Ajustes para caber na GTX 1650 ---
        imgsz=512,           # 640 é muito alto para 4GB; 512 é ideal
        batch=4,             # valor realista (4 → pode falhar, 2 é seguro)
        workers=1,           # Windows + VRAM baixa = 1 worker é mais estável

        epochs=200,
        cache="disk",
        patience=60,
        device=0 if torch.cuda.is_available() else "cpu",
        project=project,
        name=f"whitefly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

        # --- Treino estável ---
        optimizer="AdamW",
        lr0=0.001,
        amp=False,           # IMPORTANTÍSSIMO: 1650 gera NaN com AMP
        half=False,          # FP16 causa instabilidade em GPUs 4GB
        single_cls=True,

        # --- Augmentations balanceadas ---
        multi_scale=True,        # ✅ Escalas variadas
        mosaic=0.5,              # ✅ Contexto rico
        close_mosaic=150,        # ✅ ADICIONE - Ajuste fino no final
        mixup=0.0,               # ✅ OK por enquanto
        auto_augment=None,       # ✅ Mantém controle manual

        # --- Mantém leves ---
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        erasing=0.1,

        # --- HSV otimizado ---
        hsv_h=0.017,
        hsv_s=0.5,
        hsv_v=0.4,

        # --- Outros ---
        overlap_mask=True,
        save_period=-1,
        verbose=True,
        plots=False
    )
    # set model path
    args["model"] = f"{model_version}.pt"
    return args
def convert_labels_to_single_class(dataset_root: Path, target_orig_class: int = TARGET_ORIG_CLASS):
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
                # NÃO deletar! Apenas manter vazio.
                txt_path.write_text("", encoding="utf-8")
                removed += 1

    print(f"Conversão concluída. Mantidos: {kept}, esvaziados: {removed}.")


def main():
    print("\n=== Iniciando preparação e treino (whitefly) ===\n")

    # Confirmar YAML (mantendo caminho como o usuário pediu)
    yaml_path = DATASET_YAML_PATH
    if not yaml_path.exists():
        print("Erro: YAML não encontrado:", yaml_path)
        sys.exit(1)

    # Preparar dataset: checar/convert labels e garantir val
    dataset_root = verify_and_prepare_dataset(yaml_path)

    # Converter (se necessário) — função internamente faz backup
    # (a conversão já é chamada dentro verify_and_prepare_dataset quando detecta classes != {0})
    print("\nDataset preparado em:", dataset_root)

    # Carregar modelo
    model_version = DEFAULT_MODEL
    model = YOLO(f"{model_version}.pt")
    print("Modelo carregado:", model_version)

    # montar args de treino
    train_args = build_train_args(yaml_path, model_version=model_version, project=DEFAULT_PROJECT)

    # map para ultralytics: 'model' deve ser passado via método se não usar model.train(model=..), porém o método YOLO(...).train já usa pesos do objeto
    # aqui chamaremos try_train_with_retries com o objeto 'model' e args
    # Remover 'model' do args pois usamos model.train()
    if "model" in train_args:
        train_args.pop("model")

    # limpar VRAM antes de começar
    torch.cuda.empty_cache()

    try:
        results = try_train_with_retries(model, train_args, max_retries=MAX_RETRIES_ON_OOM)
    except Exception as e:
        print("Erro durante o treinamento:", e)
        raise

    print("\nTreinamento finalizado. Resultado salvo em:", results)
    # validação final automática
    try:
        print("\nExecutando validação final (self.model.val)...")
        model.val(plots=True)
    except Exception as e:
        print("Validação final falhou:", e)

    # limpar VRAM no final
    torch.cuda.empty_cache()
    print("\n=== Processo concluído ===\n")


if __name__ == "__main__":
    main()
