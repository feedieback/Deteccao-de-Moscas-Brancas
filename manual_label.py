# -*- coding: utf-8 -*-
"""
Ferramenta de Refinamento Manual para Auto-Labeling
Interface interativa para revisar, corrigir e adicionar detecções
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import shutil


class ManualRefinementTool:
    """Ferramenta interativa para refinamento manual"""
    
    def __init__(self, dataset_root: Path):
        self.selected_bboxes = set()   # múltiplas caixas selecionadas
        self.selection_area_start = None
        self.selection_area_temp = None

        self.dataset_root = Path(dataset_root)
        self.current_split = 'train'
        self.current_index = 0
        self.images = []
        self.current_image = None
        self.current_labels = []
        self.display_image = None
        self.scale = 1.0
        
        # Estados
        self.drawing_bbox = False
        self.bbox_start = None
        self.temp_bbox = None
        self.selected_bbox_idx = None
        self.deleted_boxes = []
        
        # Estatísticas
        self.stats = {
            'reviewed': 0,
            'added': 0,
            'deleted': 0,
            'modified': 0,
            'skipped': 0
        }
        
        # Configurações visuais
        self.colors = {
            'existing': (0, 255, 0),      # Verde: detecção existente
            'selected': (0, 255, 255),    # Amarelo: selecionada
            'drawing': (255, 0, 255),     # Magenta: desenhando
            'new': (255, 165, 0)          # Laranja: nova detecção
        }
        
        print("="*80)
        print(" "*15 + "FERRAMENTA DE REFINAMENTO MANUAL")
        print("="*80)
        print("\nControles:")
        print("  MOUSE:")
        print("    • Clique esquerdo + arraste: Desenhar nova bbox")
        print("    • Clique direito: Selecionar/deselecionar bbox")
        print("  ")
        print("  TECLADO:")
        print("    [ESPAÇO] - Próxima imagem (salvar alterações)")
        print("    [D] - Deletar bbox selecionada")
        print("    [U] - Desfazer última deleção")
        print("    [R] - Resetar imagem (descartar alterações)")
        print("    [S] - Pular imagem sem salvar")
        print("    [C] - Limpar todas as detecções")
        print("    [A] - Aceitar imagem e avançar")
        print("    [+/-] - Zoom in/out")
        print("    [Q] - Salvar e sair")
        print("    [ESC] - Sair sem salvar")
        print("\n" + "="*80)
    
    def load_split_images(self, split: str):
        """Carrega imagens do split"""
        
        img_dir = self.dataset_root / 'images' / split
        
        if not img_dir.exists():
            print(f"❌ Diretório não encontrado: {img_dir}")
            return False
        
        self.images = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        self.current_split = split
        self.current_index = 0
        
        print(f"\n✓ Carregadas {len(self.images)} imagens do split '{split}'")
        return len(self.images) > 0
    
    def load_current_image(self):
        """Carrega imagem atual e suas labels"""
        
        if self.current_index >= len(self.images):
            return False
        
        img_path = self.images[self.current_index]
        self.current_image = cv2.imread(str(img_path))
        
        if self.current_image is None:
            print(f"❌ Erro ao carregar: {img_path}")
            return False
        
        # Carregar labels
        label_path = self.dataset_root / 'labels' / self.current_split / f"{img_path.stem}.txt"
        self.current_labels = self.load_yolo_labels(label_path)
        
        # Resetar estados
        self.selected_bbox_idx = None
        self.deleted_boxes = []
        
        return True
    
    def load_yolo_labels(self, label_path: Path) -> List[Dict]:
        """Carrega labels YOLO e converte para formato interno"""
        
        if not label_path.exists():
            return []
        
        labels = []
        h, w = self.current_image.shape[:2]
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    cls, x_center, y_center, width, height = map(float, parts)
                    
                    # Converter para coordenadas absolutas
                    x = int((x_center - width / 2) * w)
                    y = int((y_center - height / 2) * h)
                    bw = int(width * w)
                    bh = int(height * h)
                    
                    labels.append({
                        'class': int(cls),
                        'bbox': [x, y, bw, bh],
                        'modified': False,
                        'is_new': False
                    })
        except Exception as e:
            print(f"⚠️ Erro ao ler {label_path}: {e}")
        
        return labels
    
    def save_yolo_labels(self, label_path: Path):
        """Salva labels no formato YOLO"""
        
        h, w = self.current_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for label in self.current_labels:
                x, y, bw, bh = label['bbox']
                
                # Converter para formato YOLO
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                width = bw / w
                height = bh / h
                
                # Clipar valores
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0, 1)
                height = np.clip(height, 0, 1)
                
                cls = label['class']
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def draw_interface(self):
        """Desenha a interface com todas as informações"""
        
        # Copiar imagem
        self.display_image = self.current_image.copy()
        h, w = self.display_image.shape[:2]
        
        # Desenhar bboxes existentes
        for idx, label in enumerate(self.current_labels):
            x, y, bw, bh = label['bbox']
            
            # Cor baseada no estado
            if idx in self.selected_bboxes:
                color = self.colors['selected']
                thickness = 3

            elif label.get('is_new', False):
                color = self.colors['new']
                thickness = 2
            else:
                color = self.colors['existing']
                thickness = 2
            
            cv2.rectangle(self.display_image, (x, y), (x + bw, y + bh), color, thickness)
            
            # Label
            label_text = f"#{idx+1}"
            if label.get('is_new'):
                label_text += " NEW"
            
            cv2.putText(self.display_image, label_text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Desenhar bbox temporária (enquanto desenha)
        if self.temp_bbox is not None:
            x1, y1, x2, y2 = self.temp_bbox
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), 
                         self.colors['drawing'], 2)
            
        if self.selection_area_temp is not None:
            x1,y1,x2,y2 = self.selection_area_temp
            cv2.rectangle(self.display_image,
                        (x1, y1), (x2, y2),
                        (255, 255, 0), 2)
        
        # Informações no topo
        info_bg = np.zeros((100, w, 3), dtype=np.uint8)
        
        # Linha 1: Imagem atual
        img_name = self.images[self.current_index].name
        text1 = f"Imagem: {self.current_index + 1}/{len(self.images)} - {img_name}"
        cv2.putText(info_bg, text1, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Linha 2: Detecções
        text2 = f"Deteccoes: {len(self.current_labels)}"
        if self.selected_bbox_idx is not None:
            text2 += f" | Selecionada: #{self.selected_bbox_idx + 1}"
        cv2.putText(info_bg, text2, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Linha 3: Estatísticas
        text3 = f"Revisadas: {self.stats['reviewed']} | Adicionadas: {self.stats['added']} | Deletadas: {self.stats['deleted']}"
        cv2.putText(info_bg, text3, (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Combinar
        self.display_image = np.vstack([info_bg, self.display_image])
        
        # Legenda lateral
        legend_w = 250
        legend = np.zeros((h + 100, legend_w, 3), dtype=np.uint8)
        
        y_pos = 30
        cv2.putText(legend, "LEGENDA:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 40
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), self.colors['existing'], -1)
        cv2.putText(legend, "Existente", (40, y_pos + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 35
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), self.colors['new'], -1)
        cv2.putText(legend, "Nova", (40, y_pos + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 35
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), self.colors['selected'], -1)
        cv2.putText(legend, "Selecionada", (40, y_pos + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 50
        cv2.putText(legend, "ATALHOS:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        shortcuts = [
            "SPACE: Salvar",
            "D: Deletar",
            "U: Desfazer",
            "R: Resetar",
            "C: Limpar",
            "A: Aceitar",
            "Q: Sair"
        ]
        
        y_pos += 30
        for shortcut in shortcuts:
            cv2.putText(legend, shortcut, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 25
        
        # Combinar com legenda
        self.display_image = np.hstack([self.display_image, legend])
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback do mouse"""

        y_adjusted = y - 100
        if y_adjusted < 0:
            return

        # =========================
        #  DESENHO DE NOVA BBOX
        # =========================
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_bbox = True
            self.bbox_start = (x, y_adjusted)
            self.temp_bbox = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_bbox:
            if self.bbox_start is not None:
                self.temp_bbox = (self.bbox_start[0], self.bbox_start[1], x, y_adjusted)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing_bbox and self.bbox_start is not None:
                x1, y1 = self.bbox_start
                x2, y2 = x, y_adjusted
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                bw = x2 - x1
                bh = y2 - y1

                if bw > 10 and bh > 10:
                    self.current_labels.append({
                        'class': 0,
                        'bbox': [x1, y1, bw, bh],
                        'modified': True,
                        'is_new': True
                    })
                    print(f"✓ Nova bbox adicionada")

                self.drawing_bbox = False
                self.bbox_start = None
                self.temp_bbox = None

        # =========================
        #   SELEÇÃO MÚLTIPLA / ÁREA
        # =========================
        if event == cv2.EVENT_RBUTTONDOWN:
            self.selection_area_start = (x, y_adjusted)
            self.selection_area_temp = None

        elif event == cv2.EVENT_MOUSEMOVE and self.selection_area_start:
            self.selection_area_temp = (
                self.selection_area_start[0],
                self.selection_area_start[1],
                x,
                y_adjusted
            )

        elif event == cv2.EVENT_RBUTTONUP:
            if self.selection_area_start is None:
                return

            x1, y1 = self.selection_area_start
            x2, y2 = x, y_adjusted
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Clique curto → seleciona apenas aquela bbox
            if abs(x2 - x1) < 10 and abs(y2 - y1) < 10:
                clicked = False
                for idx, label in enumerate(self.current_labels):
                    bx, by, bw, bh = label['bbox']
                    if bx <= x <= bx + bw and by <= y_adjusted <= by + bh:
                        if idx in self.selected_bboxes:
                            self.selected_bboxes.remove(idx)
                            print(f"◼ Caixa {idx+1} desmarcada")
                        else:
                            self.selected_bboxes.add(idx)
                            print(f"✓ Caixa {idx+1} selecionada")
                        clicked = True
                        break

                # Clique no vazio → remove seleções
                if not clicked:
                    self.selected_bboxes.clear()

            # Seleção por área (marquee)
            else:
                count = 0
                for idx, label in enumerate(self.current_labels):
                    bx, by, bw, bh = label['bbox']
                    if bx >= x1 and by >= y1 and (bx + bw) <= x2 and (by + bh) <= y2:
                        self.selected_bboxes.add(idx)
                        count += 1

                print(f"✓ {count} caixas selecionadas pela área")

            self.selection_area_start = None
            self.selection_area_temp = None

    def delete_selected_bbox(self):
        if not self.selected_bboxes:
            return

        for idx in sorted(self.selected_bboxes, reverse=True):
            deleted = self.current_labels.pop(idx)
            self.deleted_boxes.append((idx, deleted))
            self.stats['deleted'] += 1
            print(f"✓ Bbox #{idx+1} deletada")

        self.selected_bboxes.clear()

    def undo_delete(self):
        """Desfaz última deleção"""
        
        if self.deleted_boxes:
            idx, label = self.deleted_boxes.pop()
            self.current_labels.insert(idx, label)
            self.stats['deleted'] -= 1
            print(f"✓ Deleção desfeita")
    
    def clear_all_labels(self):
        """Limpa todas as labels"""
        
        if self.current_labels:
            for label in self.current_labels:
                self.deleted_boxes.append((0, label))
            
            count = len(self.current_labels)
            self.current_labels = []
            self.stats['deleted'] += count
            print(f"✓ {count} labels removidas")
    
    def reset_image(self):
        """Reseta imagem para estado original"""
        
        self.load_current_image()
        print(f"✓ Imagem resetada")
    
    def save_and_next(self):
        """Salva alterações e vai para próxima"""
        
        # Salvar labels
        label_path = self.dataset_root / 'labels' / self.current_split / f"{self.images[self.current_index].stem}.txt"
        self.save_yolo_labels(label_path)
        
        self.stats['reviewed'] += 1
        print(f"✓ Salvo: {len(self.current_labels)} detecções")
        
        # Próxima imagem
        self.current_index += 1
        
        if self.current_index < len(self.images):
            self.load_current_image()
            return True
        else:
            return False
    
    def skip_image(self):
        """Pula imagem sem salvar"""
        
        self.stats['skipped'] += 1
        self.current_index += 1
        
        if self.current_index < len(self.images):
            self.load_current_image()
            return True
        else:
            return False
    
    def run(self):
        """Loop principal"""
        
        # Escolher split
        print("\nEscolha o split para revisar:")
        print("  1. train")
        print("  2. val")
        print("  3. test")
        
        choice = input("\nOpção (1-3): ")
        
        split_map = {'1': 'train', '2': 'val', '3': 'test'}
        split = split_map.get(choice, 'train')
        
        if not self.load_split_images(split):
            return
        
        # Carregar primeira imagem
        if not self.load_current_image():
            print("❌ Erro ao carregar primeira imagem")
            return
        
        # Criar backup
        backup_dir = self.dataset_root / 'labels_manual_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        labels_dir = self.dataset_root / 'labels'
        
        print(f"\n📦 Criando backup em: {backup_dir.name}")
        shutil.copytree(labels_dir, backup_dir)
        
        # Criar janela
        window_name = 'Refinamento Manual - Mosca-Branca'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"\n▶ Iniciando revisão de {len(self.images)} imagens")
        print("   Pressione 'H' para ver ajuda")
        
        running = True
        
        while running:
            self.draw_interface()
            cv2.imshow(window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Espaço: Salvar e próxima
            if key == ord(' '):
                if not self.save_and_next():
                    print("\n✓ Última imagem revisada!")
                    running = False
            
            # D: Deletar selecionada
            elif key == ord('d') or key == ord('D'):
                self.delete_selected_bbox()
            
            # U: Desfazer
            elif key == ord('u') or key == ord('U'):
                self.undo_delete()
            
            # R: Resetar
            elif key == ord('r') or key == ord('R'):
                self.reset_image()
            
            # C: Limpar tudo
            elif key == ord('c') or key == ord('C'):
                confirm = input("\n⚠️ Limpar TODAS as detecções? (s/n): ")
                if confirm.lower() == 's':
                    self.clear_all_labels()
            
            # A: Aceitar e avançar
            elif key == ord('a') or key == ord('A'):
                if not self.save_and_next():
                    running = False
            
            # S: Pular
            elif key == ord('s') or key == ord('S'):
                if not self.skip_image():
                    running = False
            
            # Q: Sair
            elif key == ord('q') or key == ord('Q'):
                confirm = input("\n⚠️ Salvar alterações e sair? (s/n): ")
                if confirm.lower() == 's':
                    label_path = self.dataset_root / 'labels' / self.current_split / f"{self.images[self.current_index].stem}.txt"
                    self.save_yolo_labels(label_path)
                running = False
            
            # ESC: Cancelar
            elif key == 27:
                confirm = input("\n⚠️ Sair SEM salvar? (s/n): ")
                if confirm.lower() == 's':
                    running = False
            
            # H: Ajuda
            elif key == ord('h') or key == ord('H'):
                print("\n" + "="*60)
                print("AJUDA - CONTROLES")
                print("="*60)
                print("MOUSE:")
                print("  • Clique esquerdo + arraste: Nova bbox")
                print("  • Clique direito: Selecionar bbox")
                print("\nTECLADO:")
                print("  ESPAÇO - Salvar e próxima")
                print("  D - Deletar selecionada")
                print("  U - Desfazer deleção")
                print("  R - Resetar imagem")
                print("  C - Limpar todas")
                print("  A - Aceitar e avançar")
                print("  S - Pular sem salvar")
                print("  Q - Salvar e sair")
                print("  ESC - Sair sem salvar")
                print("="*60)
        
        cv2.destroyAllWindows()
        
        # Resumo final
        self.print_final_summary()
    
    def print_final_summary(self):
        """Imprime resumo final"""
        
        print("\n" + "="*80)
        print("RESUMO DA REVISÃO MANUAL")
        print("="*80)
        
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   Imagens revisadas: {self.stats['reviewed']}")
        print(f"   Detecções adicionadas: {self.stats['added']}")
        print(f"   Detecções deletadas: {self.stats['deleted']}")
        print(f"   Imagens puladas: {self.stats['skipped']}")
        
        print(f"\n✅ Revisão concluída!")
        print(f"   Dataset: {self.dataset_root}")
        print(f"   Split: {self.current_split}")


def main():
    print("\n" + "="*80)
    print(" "*12 + "FERRAMENTA DE REFINAMENTO MANUAL")
    print(" "*18 + "Mosca-Branca Dataset")
    print("="*80)
    
    dataset_root = Path(r"C:\Users\Victor\Documents\TCC\IA\datasets\ip102_yolo_white_fly")
    
    if not dataset_root.exists():
        print(f"\n❌ Dataset não encontrado: {dataset_root}")
        input("\nPressione Enter para sair...")
        return
    
    print(f"\n✓ Dataset encontrado: {dataset_root}")
    
    print("\n💡 Esta ferramenta permite:")
    print("  • Revisar detecções automáticas")
    print("  • Adicionar detecções perdidas")
    print("  • Remover falsos positivos")
    print("  • Corrigir bounding boxes")
    
    confirm = input("\nContinuar? (s/n): ")
    if confirm.lower() != 's':
        return
    
    tool = ManualRefinementTool(dataset_root)
    tool.run()
    
    input("\nPressione Enter para sair...")


if __name__ == "__main__":
    main()