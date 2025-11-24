# -*- coding: utf-8 -*-
"""
Ferramenta de Refinamento Manual para Auto-Labeling
Interface gráfica para revisar, corrigir e adicionar bounding boxes geradas automaticamente.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import shutil


class ManualRefinementTool:
    """
    Ferramenta interativa para revisar e ajustar labels YOLO.
    Permite visualizar detecções, criar novas caixas, remover, selecionar múltiplas,
    e salvar as correções diretamente no dataset.
    """
    
    def __init__(self, dataset_root: Path):
        # Armazena múltiplas caixas selecionadas simultaneamente
        self.selected_bboxes = set()

        # Coordenadas temporárias de seleção retangular
        self.selection_area_start = None
        self.selection_area_temp = None

        # Informações do dataset e estado da interface
        self.dataset_root = Path(dataset_root)
        self.current_split = 'train'
        self.current_index = 0
        self.images = []
        self.current_image = None
        self.current_labels = []
        self.display_image = None
        self.scale = 1.0
        
        # Estados de desenho e seleção
        self.drawing_bbox = False
        self.bbox_start = None
        self.temp_bbox = None
        self.selected_bbox_idx = None
        self.deleted_boxes = []
        
        # Estatísticas da sessão
        self.stats = {
            'reviewed': 0,    # imagens revisadas
            'added': 0,       # novas caixas criadas
            'deleted': 0,     # caixas removidas
            'modified': 0,    # caixas ajustadas
            'skipped': 0      # imagens puladas
        }
        
        # Cores utilizadas para destacar caixas
        self.colors = {
            'existing': (0, 255, 0),      # caixa existente (verde)
            'selected': (0, 255, 255),    # caixa selecionada (amarelo)
            'drawing': (255, 0, 255),     # caixa sendo desenhada (magenta)
            'new': (255, 165, 0)          # nova caixa adicionada (laranja)
        }
        
        # Instruções exibidas no terminal ao iniciar
        print("="*80)
        print(" "*15 + "FERRAMENTA DE REFINAMENTO MANUAL")
        print("="*80)
        print("\nControles:")
        print("  MOUSE:")
        print("    • Clique esquerdo + arrastar = desenhar nova bounding box")
        print("    • Clique direito = selecionar / deselecionar caixas")
        print("\n  TECLADO:")
        print("    [ESPAÇO] - salvar alterações e avançar")
        print("    [D] - deletar caixas selecionadas")
        print("    [U] - desfazer última deleção")
        print("    [R] - recarregar imagem e descartar alterações")
        print("    [S] - pular imagem sem salvar")
        print("    [C] - remover todas as detecções")
        print("    [A] - aceitar imagem atual e avançar")
        print("    [+/-] - zoom")
        print("    [Q] - salvar e sair")
        print("    [ESC] - sair sem salvar")
        print("="*80)
    
    def load_split_images(self, split: str):
        """
        Carrega todas as imagens do split selecionado (train/val/test).
        Retorna True caso haja imagens disponíveis.
        """
        
        img_dir = self.dataset_root / 'images' / split
        
        if not img_dir.exists():
            print(f"❌ Diretório não encontrado: {img_dir}")
            return False
        
        # Lista todos os arquivos JPG/PNG
        self.images = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        self.current_split = split
        self.current_index = 0
        
        print(f"\n✓ {len(self.images)} imagens carregadas do split '{split}'")
        return len(self.images) > 0
    
    def load_current_image(self):
        """
        Carrega a imagem atual e lê suas labels YOLO.
        Também reseta indicadores de seleção.
        """
        
        if self.current_index >= len(self.images):
            return False
        
        img_path = self.images[self.current_index]
        self.current_image = cv2.imread(str(img_path))
        
        if self.current_image is None:
            print(f"❌ Erro ao carregar: {img_path}")
            return False
        
        # Carrega labels correspondentes
        label_path = self.dataset_root / 'labels' / self.current_split / f"{img_path.stem}.txt"
        self.current_labels = self.load_yolo_labels(label_path)
        
        # Limpa estados antigos
        self.selected_bbox_idx = None
        self.deleted_boxes = []
        
        return True
    
    def load_yolo_labels(self, label_path: Path) -> List[Dict]:
        """
        Lê um arquivo .txt no formato YOLO e converte para coordenadas absolutas.
        Cada label retorna: {class, bbox[x,y,w,h], modified, is_new}
        """
        
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
                    
                    # Conversão YOLO → coordenadas absolutas
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
        """
        Converte as bounding boxes ajustadas para o formato YOLO
        e salva no arquivo TXT correspondente.
        """
        
        h, w = self.current_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for label in self.current_labels:
                x, y, bw, bh = label['bbox']
                
                # Converte coordenadas absolutas para YOLO normalizado
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                width = bw / w
                height = bh / h
                
                # Garante que valores fiquem entre 0 e 1
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0, 1)
                height = np.clip(height, 0, 1)
                
                f.write(f"{label['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def draw_interface(self):
        """
        Renderiza a interface gráfica:
        - imagem original
        - bounding boxes coloridas
        - painel informativo superior
        - legenda lateral de cores
        """
        
        # Cria cópia da imagem exibida
        self.display_image = self.current_image.copy()
        h, w = self.display_image.shape[:2]
        
        # Desenha todas as bounding boxes carregadas
        for idx, label in enumerate(self.current_labels):
            x, y, bw, bh = label['bbox']
            
            # Escolhe cor baseado no estado da caixa
            if idx in self.selected_bboxes:
                color = self.colors['selected']
                thickness = 3
            elif label.get('is_new', False):
                color = self.colors['new']
                thickness = 2
            else:
                color = self.colors['existing']
                thickness = 2
            
            # Desenha retângulo
            cv2.rectangle(self.display_image, (x, y), (x + bw, y + bh), color, thickness)
            
            # Identificação textual
            tag = f"#{idx+1}"
            if label.get('is_new'):
                tag += " NEW"
            
            cv2.putText(self.display_image, tag, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Desenha caixa temporária enquanto o usuário arrasta o mouse
        if self.temp_bbox is not None:
            x1, y1, x2, y2 = self.temp_bbox
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2),
                         self.colors['drawing'], 2)
            
        # Adiciona área de seleção múltipla (marquee)
        if self.selection_area_temp is not None:
            x1,y1,x2,y2 = self.selection_area_temp
            cv2.rectangle(self.display_image,
                        (x1, y1), (x2, y2),
                        (255, 255, 0), 2)
        
        # Painel superior com informações da imagem atual
        info_bg = np.zeros((100, w, 3), dtype=np.uint8)
        
        text1 = f"Imagem {self.current_index + 1}/{len(self.images)} - {self.images[self.current_index].name}"
        cv2.putText(info_bg, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        text2 = f"Detecções: {len(self.current_labels)}"
        if self.selected_bboxes:
            text2 += f" | Selecionadas: {len(self.selected_bboxes)}"
        cv2.putText(info_bg, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        
        text3 = f"Revisadas: {self.stats['reviewed']} | Adicionadas: {self.stats['added']} | Deletadas: {self.stats['deleted']}"
        cv2.putText(info_bg, text3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Integra painel à imagem principal
        self.display_image = np.vstack([info_bg, self.display_image])
        
        # Cria legenda lateral
        legend_w = 250
        legend = np.zeros((h + 100, legend_w, 3), dtype=np.uint8)
        
        y_pos = 30
        cv2.putText(legend, "LEGENDA:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        y_pos += 40
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), self.colors['existing'], -1)
        cv2.putText(legend, "Caixa existente", (40, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        y_pos += 35
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), self.colors['new'], -1)
        cv2.putText(legend, "Nova caixa", (40, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        y_pos += 35
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), self.colors['selected'], -1)
        cv2.putText(legend, "Selecionada", (40, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        y_pos += 50
        cv2.putText(legend, "ATALHOS:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Lista de comandos básicos
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
        for s in shortcuts:
            cv2.putText(legend, s, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 25
        
        # Une imagem e legenda lateral
        self.display_image = np.hstack([self.display_image, legend])
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Lógica principal de interação do mouse:
        - desenhar nova caixa (botão esquerdo)
        - selecionar múltiplas caixas ou uma única (botão direito)
        """
        y_adjusted = y - 100  # Desconta a área de informações
        if y_adjusted < 0:
            return

        # ---------------------------
        #  DESENHO DE NOVA BBOX
        # ---------------------------
        if event == cv2.EVENT_LBUTTONDOWN:
            # Inicia criação da caixa
            self.drawing_bbox = True
            self.bbox_start = (x, y_adjusted)
            self.temp_bbox = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_bbox:
            # Atualiza caixa enquanto arrasta o mouse
            x1, y1 = self.bbox_start
            self.temp_bbox = (x1, y1, x, y_adjusted)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finaliza caixa criada
            if self.drawing_bbox and self.bbox_start is not None:
                x1, y1 = self.bbox_start
                x2, y2 = x, y_adjusted
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                bw = x2 - x1
                bh = y2 - y1

                # Ignora caixas minúsculas
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

        # ---------------------------
        #  SELEÇÃO POR ÁREA / CLIQUE
        # ---------------------------
        if event == cv2.EVENT_RBUTTONDOWN:
            # Inicia seleção retangular
            self.selection_area_start = (x, y_adjusted)
            self.selection_area_temp = None

        elif event == cv2.EVENT_MOUSEMOVE and self.selection_area_start:
            # Atualiza seleção retangular durante arrasto
            sx, sy = self.selection_area_start
            self.selection_area_temp = (sx, sy, x, y_adjusted)

        elif event == cv2.EVENT_RBUTTONUP:
            # Finaliza seleção retangular
            if self.selection_area_start is None:
                return

            x1, y1 = self.selection_area_start
            x2, y2 = x, y_adjusted
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Clique curto → seleciona uma única caixa
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

                if not clicked:
                    # Clique no vazio limpa seleção
                    self.selected_bboxes.clear()

            # Seleção retangular → seleciona múltiplas
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
        """
        Remove todas as bounding boxes atualmente selecionadas.
        Armazena deletadas para permitir desfazer (undo).
        """
        if not self.selected_bboxes:
            return

        for idx in sorted(self.selected_bboxes, reverse=True):
            deleted = self.current_labels.pop(idx)
            self.deleted_boxes.append((idx, deleted))
            self.stats['deleted'] += 1
            print(f"✓ Caixa #{idx+1} deletada")

        self.selected_bboxes.clear()

    def undo_delete(self):
        """
        Restaura a última caixa deletada (função undo).
        """
        if self.deleted_boxes:
            idx, label = self.deleted_boxes.pop()
            self.current_labels.insert(idx, label)
            self.stats['deleted'] -= 1
            print(f"✓ Deleção desfeita")
    
    def clear_all_labels(self):
        """
        Remove todas as detecções da imagem atual.
        Todas são armazenadas para permitir desfazer.
        """
        if self.current_labels:
            for label in self.current_labels:
                self.deleted_boxes.append((0, label))
            
            count = len(self.current_labels)
            self.current_labels = []
            self.stats['deleted'] += count
            print(f"✓ {count} labels removidas")
    
    def reset_image(self):
        """
        Recarrega a imagem e suas labels originais, descartando todas
        as alterações realizadas na sessão atual.
        """
        self.load_current_image()
        print(f"✓ Imagem resetada")
    
    def save_and_next(self):
        """
        Salva todas as labels da imagem atual e avança para a próxima imagem.
        Retorna False quando chega ao final da lista de imagens.
        """
        label_path = self.dataset_root / 'labels' / self.current_split / f"{self.images[self.current_index].stem}.txt"
        self.save_yolo_labels(label_path)
        
        self.stats['reviewed'] += 1
        print(f"✓ Salvo: {len(self.current_labels)} detecções")
        
        self.current_index += 1
        
        if self.current_index < len(self.images):
            self.load_current_image()
            return True
        else:
            return False
    
    def skip_image(self):
        """
        Pula a imagem atual sem salvar qualquer modificação.
        """
        self.stats['skipped'] += 1
        self.current_index += 1
        
        if self.current_index < len(self.images):
            self.load_current_image()
            return True
        else:
            return False
    
    def run(self):
        """
        Loop principal da ferramenta.
        Gerencia entradas do teclado, mouse, salvamento e renderização da interface.
        """
        
        # Seleção do conjunto train/val/test
        print("\nEscolha o split para revisar:")
        print("  1. train")
        print("  2. val")
        print("  3. test")
        
        choice = input("\nOpção (1-3): ")
        
        split_map = {'1': 'train', '2': 'val', '3': 'test'}
        split = split_map.get(choice, 'train')
        
        if not self.load_split_images(split):
            return
        
        # Carrega primeira imagem
        if not self.load_current_image():
            print("❌ Erro ao carregar a primeira imagem")
            return
        
        # Cria backup completo das labels antes da edição
        backup_dir = self.dataset_root / 'labels_manual_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        labels_dir = self.dataset_root / 'labels'
        
        print(f"\n📦 Criando backup em: {backup_dir.name}")
        shutil.copytree(labels_dir, backup_dir)
        
        # Inicializa janela gráfica
        window_name = 'Refinamento Manual - Mosca-Branca'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"\n▶ Iniciando revisão de {len(self.images)} imagens")
        print("   Pressione 'H' para ajuda")
        
        running = True
        
        while running:
            self.draw_interface()
            cv2.imshow(window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            # ---- TECLAS ----
            if key == ord(' '):  # salvar e ir para próxima
                if not self.save_and_next():
                    print("\n✓ Última imagem revisada!")
                    running = False
            
            elif key in (ord('d'), ord('D')):   # deletar selecionada
                self.delete_selected_bbox()
            
            elif key in (ord('u'), ord('U')):   # desfazer deleção
                self.undo_delete()
            
            elif key in (ord('r'), ord('R')):   # resetar imagem
                self.reset_image()
            
            elif key in (ord('c'), ord('C')):   # limpar tudo
                confirm = input("\n⚠️ Remover TODAS as detecções? (s/n): ")
                if confirm.lower() == 's':
                    self.clear_all_labels()
            
            elif key in (ord('a'), ord('A')):   # aceitar e avançar
                if not self.save_and_next():
                    running = False
            
            elif key in (ord('s'), ord('S')):   # pular imagem
                if not self.skip_image():
                    running = False
            
            elif key in (ord('q'), ord('Q')):   # sair salvando
                confirm = input("\n⚠️ Salvar alterações e sair? (s/n): ")
                if confirm.lower() == 's':
                    label_path = self.dataset_root / 'labels' / self.current_split / f"{self.images[self.current_index].stem}.txt"
                    self.save_yolo_labels(label_path)
                running = False
            
            elif key == 27:  # ESC — sair sem salvar
                confirm = input("\n⚠️ Sair SEM salvar? (s/n): ")
                if confirm.lower() == 's':
                    running = False
            
            elif key in (ord('h'), ord('H')):  # ajuda no terminal
                print("\n" + "="*60)
                print("AJUDA - CONTROLES")
                print("="*60)
                print("MOUSE:")
                print("  • Clique esquerdo + arraste = nova bbox")
                print("  • Clique direito = selecionar/ou desmarcar bbox")
                print("\nTECLADO:")
                print("  ESPAÇO = salvar e avançar")
                print("  D = deletar caixa selecionada")
                print("  U = desfazer")
                print("  R = resetar imagem")
                print("  C = limpar todas as caixas")
                print("  A = aceitar e avançar")
                print("  S = pular sem salvar")
                print("  Q = aceitar e sair")
                print("  ESC = sair sem salvar")
                print("="*60)
        
        cv2.destroyAllWindows()
        
        # Exibe estatísticas finais
        self.print_final_summary()
    
    def print_final_summary(self):
        """
        Mostra no console um resumo da sessão de revisão.
        """
        
        print("\n" + "="*80)
        print("RESUMO DA REVISÃO MANUAL")
        print("="*80)
        
        print("\n📊 ESTATÍSTICAS:")
        print(f"   Imagens revisadas: {self.stats['reviewed']}")
        print(f"   Detecções adicionadas: {self.stats['added']}")
        print(f"   Detecções deletadas: {self.stats['deleted']}")
        print(f"   Imagens puladas: {self.stats['skipped']}")
        
        print("\n✅ Revisão concluída!")
        print(f"Dataset: {self.dataset_root}")
        print(f"Split revisado: {self.current_split}")


def main():
    """
    Função principal:
    - verifica o dataset
    - exibe instruções iniciais
    - inicia a ferramenta gráfica
    """
    
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
    print("  • Adicionar caixas não detectadas")
    print("  • Remover falsos positivos")
    print("  • Corrigir bounding boxes imprecisas")
    
    confirm = input("\nContinuar? (s/n): ")
    if confirm.lower() != 's':
        return
    
    tool = ManualRefinementTool(dataset_root)
    tool.run()
    
    input("\nPressione Enter para sair...")


if __name__ == "__main__":
    main()
