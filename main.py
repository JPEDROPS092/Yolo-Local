# Filename: main_aprimorado.py

import cv2
import yaml
import numpy as np
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox
from threading import Thread, Lock
import traceback
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime
import os
import sys
from ultralytics import YOLO
from pynput import keyboard
from PIL import Image # Necessário para CTkImage
import torch # Para verificação CUDA

import logging

# Configuração do Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('navigation_aprimorado.log', mode='w'), # 'w' para sobrescrever o log a cada execução
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PREDEFINED_MODELS = {
    "YOLOv8n (Nano)": "yolov8n.pt",
    "YOLOv8s (Pequeno)": "yolov8s.pt",
    "YOLOv8m (Médio)": "yolov8m.pt",
    "YOLOv8l (Grande)": "yolov8l.pt",
    "YOLOv8x (Extra Grande)": "yolov8x.pt",
}

# --- Classe ObjectDetector (com device, iou, max_det) ---
class ObjectDetector:
    """Detecção de objetos usando YOLO."""
    def __init__(self, model_path: str, device: str = 'cpu', conf_threshold: float = 0.5, iou_threshold: float = 0.45, max_det: int = 300):
        """
        Inicializa o detector.

        Args:
            model_path (str): Caminho para o arquivo do modelo YOLO.
            device (str): Dispositivo para rodar o modelo ('cpu' ou 'cuda').
            conf_threshold (float): Limiar de confiança para detecções.
            iou_threshold (float): Limiar de IoU para Non-Maximum Suppression.
            max_det (int): Número máximo de detecções por imagem.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.device = device
        self.model_path = model_path
        self.model = None # Inicializado depois
        self._load_model()

    def _load_model(self):
        """Carrega o modelo YOLO para o dispositivo especificado."""
        try:
            logger.info(f"Tentando carregar modelo YOLO de {self.model_path} para {self.device}...")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            # Realiza uma inferência dummy para garantir que o modelo está totalmente carregado/aquecido
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)
            logger.info(f"Modelo YOLO carregado de {self.model_path} para {self.device}.")
        except FileNotFoundError:
            logger.error(f"Arquivo do modelo YOLO não encontrado: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Falha ao carregar modelo YOLO '{self.model_path}' para '{self.device}': {e}\n{traceback.format_exc()}")
            raise

    def update_parameters(self, conf_threshold: Optional[float] = None,
                          iou_threshold: Optional[float] = None,
                          max_det: Optional[int] = None):
        """Atualiza os parâmetros de detecção."""
        if conf_threshold is not None: self.conf_threshold = conf_threshold
        if iou_threshold is not None: self.iou_threshold = iou_threshold
        if max_det is not None: self.max_det = max_det
        logger.info(f"Parâmetros do detector atualizados: Conf={self.conf_threshold}, IOU={self.iou_threshold}, MaxDet={self.max_det}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Processa um frame e retorna o frame anotado e uma lista de detecções."""
        if frame is None or self.model is None:
            logger.warning("Frame None ou modelo não carregado em process_frame")
            return frame, []

        detections = []
        try:
            # Faz a predição com o modelo
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, max_det=self.max_det, verbose=False)[0]
            height, width = frame.shape[:2]
            center_x = width // 2

            # Processa os resultados
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                # A verificação de score já é feita pelo modelo com o argumento `conf`, mas uma dupla verificação não prejudica
                if score > self.conf_threshold:
                    box = [int(x1), int(y1), int(x2), int(y2)]
                    class_id = int(class_id)

                    # Calcula posição relativa ao centro
                    box_center_x = (box[0] + box[2]) // 2
                    position = "centro"
                    if box_center_x < center_x - (width * 0.1): position = "esquerda"
                    elif box_center_x > center_x + (width * 0.1): position = "direita"

                    # Calcula distância relativa baseada no tamanho da caixa
                    box_height = box[3] - box[1]
                    distance = "próximo" if box_height > height * 0.4 else \
                              "médio" if box_height > height * 0.15 else "longe"

                    # Obtém o nome da classe e desenha a caixa
                    class_name = self.model.names[class_id]
                    color = ((0, 255, 0) if score > 0.7 else (0, 255, 255) if score > 0.5 else (0, 165, 255))

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"{class_name}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    detections.append({
                        'label': class_name, 'confidence': score,
                        'position': position, 'distance': distance, 'box': box
                    })
        except Exception as e:
            logger.error(f"Erro ao processar frame: {e}\n{traceback.format_exc()}")
        return frame, detections

# --- Classe Principal da Aplicação ---
class NavigationApp:
    def __init__(self):
        self.camera = None
        self.video_capture = None # Para arquivos de vídeo
        self.detection_thread = None
        self.model_load_thread = None
        self.keyboard_listener = None
        self.window = None
        self._cleanup_called = False
        self.detector: Optional[ObjectDetector] = None
        self.video_file_path: Optional[str] = None
        self.is_video_file_mode: bool = False

        try:
            logger.info("--- Inicializando Aplicação de Navegação (Aprimorado) ---")
            self.config = self.load_config()

            self.running = False
            self.frame_lock = Lock()
            self.frame_count = 0
            self.fps = 0
            self.last_fps_update = time.time()

            self.setup_ui() # Configura a UI primeiro
            self.update_ui_from_config() # Depois popula a UI com os valores da config
            self.setup_shortcuts()

            # Tenta carregar o modelo inicial de forma não bloqueante
            self.current_model_path = self.get_model_path_from_config()
            self.current_device = "cuda" if self.config.get("enable_gpu", False) and torch.cuda.is_available() else "cpu"

            self.load_yolo_model(self.current_model_path, self.current_device, initial_load=True)

            logger.info("Aplicação inicializada com sucesso (Aprimorado)")

        except Exception as e:
            logger.critical(f"Falha crítica ao inicializar: {e}\n{traceback.format_exc()}")
            self._show_critical_error_message(f"Falha ao inicializar aplicação: {e}\nVerifique navigation_aprimorado.log.")
            self.cleanup()
            sys.exit(1)

    def _show_critical_error_message(self, message: str):
        try:
            root = ctk.CTk(); root.withdraw()
            messagebox.showerror("Erro de Inicialização", message)
            root.destroy()
        except Exception:
            print(f"CRÍTICO (sem GUI para erro): {message}", file=sys.stderr)

    def get_model_path_from_config(self) -> str:
        """Obtém o caminho do modelo a partir da configuração."""
        model_type = self.config.get("selected_model_type", "predefined")
        if model_type == "custom":
            path = self.config.get("yolo_model_path_custom", "yolov8n.pt") # Mantém nomes de arquivo .pt
        else:
            key = self.config.get("default_model_key", "YOLOv8n (Nano)") # Chave traduzida
            path = PREDEFINED_MODELS.get(key, "yolov8n.pt")

        # Garante que o caminho exista ou usa um fallback
        if not Path(path).exists():
            logger.warning(f"Caminho do modelo '{path}' da config não encontrado. Usando 'yolov8n.pt' como fallback.")
            if not Path("yolov8n.pt").exists():
                 logger.error("Modelo de fallback 'yolov8n.pt' também não encontrado. Verifique se os modelos estão disponíveis.")
                 # Idealmente, poderia baixar aqui ou levantar um erro.
                 # Por enquanto, deixará falhar durante o carregamento do modelo.
            return "yolov8n.pt"
        return path


    def load_config(self) -> Dict[str, Any]:
        """Carrega a configuração do arquivo config_aprimorado.yaml ou cria um padrão."""
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / "config_aprimorado.yaml" # Nome do arquivo de config
        default_config = {
            'camera_index': 0, 'camera_width': 1280, 'camera_height': 720, 'camera_fps': 30,
            'display_width': 800,
            'selected_model_type': "predefined", # "predefined" ou "custom"
            'default_model_key': "YOLOv8n (Nano)", # Chave de PREDEFINED_MODELS
            'yolo_model_path_custom': 'custom_model.pt', # Caminho para modelo customizado
            'enable_gpu': True, # Usar GPU
            'confidence_threshold': 0.45, 'iou_threshold': 0.5, 'max_det': 100,
            'video_loop': False # Repetir vídeo
        }

        if not config_path.exists():
            logger.warning(f"Arquivo de configuração não encontrado em {config_path}. Criando padrão.")
            try:
                with open(config_path, 'w', encoding='utf-8') as f: yaml.dump(default_config, f, sort_keys=False, allow_unicode=True)
                logger.info(f"Arquivo {config_path.name} criado com valores padrão. Revise-o.")
            except Exception as e_create:
                logger.error(f"Não foi possível criar o {config_path.name} padrão: {e_create}")
                # Continua com os padrões em memória se a criação falhar

        loaded_config = default_config
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config: # Se o arquivo não estiver vazio
                    loaded_config = {**default_config, **file_config} # Mescla, arquivo sobrescreve padrão
        except yaml.YAMLError as e:
            logger.error(f"Erro ao parsear {config_path.name}: {e}. Usando padrão.")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Erro ao carregar {config_path.name}: {e}. Usando padrão.")

        # Validações (opcional, mas boa prática)
        if not (0 <= loaded_config['camera_index'] <= 10): loaded_config['camera_index'] = 0
        if not (0.0 <= loaded_config['confidence_threshold'] <= 1.0): loaded_config['confidence_threshold'] = 0.45

        logger.info(f"Configuração carregada/definida: {loaded_config}")
        return loaded_config

    def save_config(self):
        """Salva as configurações atuais da UI no arquivo de configuração."""
        try:
            # Atualiza self.config a partir dos elementos da UI
            self.config['camera_index'] = self.camera_index_var.get()
            # self.config['camera_width'] = ... (se adicionar UI para estes)
            # self.config['camera_height'] = ...
            # self.config['camera_fps'] = ...

            self.config['selected_model_type'] = self.model_source_var.get()
            if self.model_source_var.get() == "predefined":
                self.config['default_model_key'] = self.predefined_model_var.get()
            else: # custom
                self.config['yolo_model_path_custom'] = self.custom_model_path_var.get()

            self.config['enable_gpu'] = self.gpu_var.get()
            self.config['confidence_threshold'] = self.confidence_var.get()
            self.config['iou_threshold'] = self.iou_var.get()
            self.config['max_det'] = self.max_det_var.get()
            self.config['video_loop'] = self.video_loop_var.get()

            script_dir = Path(__file__).resolve().parent
            config_path = script_dir / "config_aprimorado.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, sort_keys=False, allow_unicode=True)
            logger.info(f"Configurações salvas em {config_path}")
            self.update_status("Configurações salvas.")
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}\n{traceback.format_exc()}")
            self.show_error(f"Falha ao salvar configurações: {e}")

    def update_ui_from_config(self):
        """Popula os elementos da UI com valores de self.config."""
        self.camera_index_var.set(self.config.get('camera_index', 0))

        self.model_source_var.set(self.config.get('selected_model_type', "predefined"))
        self.predefined_model_var.set(self.config.get('default_model_key', "YOLOv8n (Nano)"))
        self.custom_model_path_var.set(self.config.get('yolo_model_path_custom', 'custom.pt'))

        self.gpu_var.set(self.config.get('enable_gpu', True))
        self.confidence_var.set(self.config.get('confidence_threshold', 0.45))
        self.iou_var.set(self.config.get('iou_threshold', 0.5))
        self.max_det_var.set(self.config.get('max_det', 100))
        self.video_loop_var.set(self.config.get('video_loop', False))

        # Atualiza visibilidade baseada na fonte do modelo
        self.on_model_source_change()
        # Atualiza estado do checkbox GPU baseado na disponibilidade de CUDA
        self.update_gpu_checkbox_state()
        # Atualiza labels dos sliders
        self.update_confidence_label()
        self.update_iou_label()
        self.update_max_det_label()

    def setup_ui(self):
        """Configura a interface gráfica do usuário."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("Assistente de Navegação AI (Aprimorado)")
        self.window.geometry("1280x850") # Altura aumentada para abas
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Container principal
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Painel esquerdo para feed de vídeo e controles principais
        left_panel = ctk.CTkFrame(main_frame, width=820) # Largura fixa para vídeo
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_panel.pack_propagate(False)

        self.video_label = ctk.CTkLabel(left_panel, text="Aguardando início...", font=ctk.CTkFont(size=20))
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)
        self.video_label.image = None

        # Barra de controle abaixo do vídeo
        control_bar = ctk.CTkFrame(left_panel, height=80)
        control_bar.pack(fill="x", pady=(5,0))

        self.toggle_button = ctk.CTkButton(control_bar, text="Iniciar Detecção", command=self.toggle_detection, height=40, font=ctk.CTkFont(size=14, weight="bold"))
        self.toggle_button.pack(side="left", padx=10, pady=10)

        self.fps_label = ctk.CTkLabel(control_bar, text="FPS: 0", width=100)
        self.fps_label.pack(side="left", padx=10, pady=10)
        self.objects_label = ctk.CTkLabel(control_bar, text="Objetos: 0", width=120)
        self.objects_label.pack(side="left", padx=10, pady=10)

        # Painel direito para abas
        right_panel = ctk.CTkFrame(main_frame, width=400)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        self.tab_view = ctk.CTkTabview(right_panel)
        self.tab_view.pack(fill="both", expand=True, padx=5, pady=5)

        tabs = ["Entrada", "Modelo", "Detecção", "Salvar"]
        for tab_name in tabs: self.tab_view.add(tab_name)

        self._create_input_tab(self.tab_view.tab("Entrada"))
        self._create_model_tab(self.tab_view.tab("Modelo"))
        self._create_detection_tab(self.tab_view.tab("Detecção"))
        self._create_save_tab(self.tab_view.tab("Salvar"))

        # Barra de status na parte inferior
        self.status_label = ctk.CTkLabel(self.window, text="Pronto.", height=30, fg_color=("gray85", "gray25"), corner_radius=6)
        self.status_label.pack(side="bottom", fill="x", padx=10, pady=(0,10))

    def _create_input_tab(self, tab):
        """Cria os controles da aba 'Entrada'."""
        ctk.CTkLabel(tab, text="Fonte de Vídeo", font=ctk.CTkFont(weight="bold")).pack(pady=10, anchor="w", padx=10)

        self.input_source_var = ctk.StringVar(value="camera") # "camera" ou "video_file"

        cam_radio = ctk.CTkRadioButton(tab, text="Câmera Ao Vivo", variable=self.input_source_var, value="camera", command=self.on_input_source_change)
        cam_radio.pack(anchor="w", padx=10, pady=5)

        cam_frame = ctk.CTkFrame(tab, fg_color="transparent")
        cam_frame.pack(fill="x", padx=20)
        ctk.CTkLabel(cam_frame, text="Índice da Câmera:").pack(side="left")
        self.camera_index_var = ctk.IntVar()
        self.camera_index_entry = ctk.CTkEntry(cam_frame, textvariable=self.camera_index_var, width=50)
        self.camera_index_entry.pack(side="left", padx=5)

        file_radio = ctk.CTkRadioButton(tab, text="Arquivo de Vídeo", variable=self.input_source_var, value="video_file", command=self.on_input_source_change)
        file_radio.pack(anchor="w", padx=10, pady=(10,5))

        self.video_file_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.video_file_frame.pack(fill="x", padx=20)
        self.video_file_button = ctk.CTkButton(self.video_file_frame, text="Carregar Vídeo", command=self.browse_video_file)
        self.video_file_button.pack(side="left")
        self.video_file_label = ctk.CTkLabel(self.video_file_frame, text="Nenhum arquivo.", wraplength=250)
        self.video_file_label.pack(side="left", padx=5)

        self.video_loop_var = ctk.BooleanVar()
        self.video_loop_checkbox = ctk.CTkCheckBox(self.video_file_frame, text="Repetir Vídeo", variable=self.video_loop_var)
        self.video_loop_checkbox.pack(side="left", padx=10)

        self.on_input_source_change() # Estado inicial

    def on_input_source_change(self):
        """Chamado quando a fonte de entrada (câmera/vídeo) muda."""
        is_video = self.input_source_var.get() == "video_file"
        self.video_file_button.configure(state="normal" if is_video else "disabled")
        self.video_file_label.configure(text_color=None if is_video else "gray") # Dica visual
        self.video_loop_checkbox.configure(state="normal" if is_video else "disabled")
        self.camera_index_entry.configure(state="normal" if not is_video else "disabled")

        if is_video:
            self.is_video_file_mode = True
            if self.video_file_path:
                 self.video_file_label.configure(text=Path(self.video_file_path).name)
            else:
                 self.video_file_label.configure(text="Nenhum arquivo selecionado.")
        else:
            self.is_video_file_mode = False

    def browse_video_file(self):
        """Abre um diálogo para selecionar um arquivo de vídeo."""
        path = filedialog.askopenfilename(
            title="Selecionar Arquivo de Vídeo",
            filetypes=(("Arquivos de Vídeo", "*.mp4 *.avi *.mov *.mkv"), ("Todos os arquivos", "*.*"))
        )
        if path:
            self.video_file_path = path
            self.video_file_label.configure(text=Path(path).name)
            logger.info(f"Arquivo de vídeo selecionado: {path}")
            self.is_video_file_mode = True # Garante que está no modo vídeo
        else:
            if not self.video_file_path: # Se nenhum caminho foi selecionado e não havia um anterior
                self.video_file_label.configure(text="Nenhum arquivo.")

    def _create_model_tab(self, tab):
        """Cria os controles da aba 'Modelo'."""
        ctk.CTkLabel(tab, text="Configuração do Modelo YOLO", font=ctk.CTkFont(weight="bold")).pack(pady=10, anchor="w", padx=10)

        self.model_source_var = ctk.StringVar(value="predefined") # "predefined" ou "custom"

        predefined_radio = ctk.CTkRadioButton(tab, text="Modelo Pré-treinado", variable=self.model_source_var, value="predefined", command=self.on_model_source_change)
        predefined_radio.pack(anchor="w", padx=10, pady=5)

        self.predefined_model_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.predefined_model_frame.pack(fill="x", padx=20)
        self.predefined_model_var = ctk.StringVar()
        self.predefined_model_combo = ctk.CTkComboBox(self.predefined_model_frame, values=list(PREDEFINED_MODELS.keys()), variable=self.predefined_model_var, command=self.on_model_selection_change)
        self.predefined_model_combo.pack(fill="x")

        custom_radio = ctk.CTkRadioButton(tab, text="Modelo Personalizado (.pt)", variable=self.model_source_var, value="custom", command=self.on_model_source_change)
        custom_radio.pack(anchor="w", padx=10, pady=(10,5))

        self.custom_model_frame = ctk.CTkFrame(tab, fg_color="transparent")
        self.custom_model_frame.pack(fill="x", padx=20)
        self.custom_model_path_var = ctk.StringVar()
        self.custom_model_entry = ctk.CTkEntry(self.custom_model_frame, textvariable=self.custom_model_path_var, state="readonly")
        self.custom_model_entry.pack(side="left", fill="x", expand=True, padx=(0,5))
        self.custom_model_browse_button = ctk.CTkButton(self.custom_model_frame, text="Buscar", width=80, command=self.browse_custom_model)
        self.custom_model_browse_button.pack(side="left")

        # Alternador de GPU
        ctk.CTkLabel(tab, text="Processamento", font=ctk.CTkFont(weight="bold")).pack(pady=(20,5), anchor="w", padx=10)
        self.gpu_var = ctk.BooleanVar()
        self.gpu_checkbox = ctk.CTkCheckBox(tab, text="Usar GPU (CUDA) se disponível", variable=self.gpu_var, command=self.on_gpu_toggle)
        self.gpu_checkbox.pack(anchor="w", padx=10)
        self.update_gpu_checkbox_state() # Define estado inicial

        # Botão Aplicar Mudanças de Modelo
        self.apply_model_button = ctk.CTkButton(tab, text="Aplicar e Recarregar Modelo", command=self.apply_model_changes)
        self.apply_model_button.pack(pady=20, padx=10, fill="x")

        self.on_model_source_change() # Define visibilidade inicial

    def update_gpu_checkbox_state(self):
        """Atualiza o estado do checkbox da GPU baseado na disponibilidade de CUDA."""
        if not torch.cuda.is_available():
            self.gpu_checkbox.configure(state="disabled", text="Usar GPU (CUDA Indisponível)")
            self.gpu_var.set(False) # Força CPU se não houver CUDA
        else:
            self.gpu_checkbox.configure(state="normal", text="Usar GPU (CUDA) se disponível")

    def on_model_source_change(self, event=None):
        """Chamado quando a fonte do modelo (pré-treinado/customizado) muda."""
        is_custom = self.model_source_var.get() == "custom"
        self.predefined_model_combo.configure(state="normal" if not is_custom else "disabled")
        # Entrada de modelo customizado: normal se custom, senão readonly
        self.custom_model_entry.configure(state="normal" if is_custom else "readonly")
        self.custom_model_browse_button.configure(state="normal" if is_custom else "disabled")
        if not is_custom: self.custom_model_entry.configure(state="readonly")


    def browse_custom_model(self):
        """Abre um diálogo para selecionar um arquivo de modelo .pt customizado."""
        path = filedialog.askopenfilename(
            title="Selecionar Modelo YOLO (.pt)",
            filetypes=(("Modelos PyTorch", "*.pt"), ("Todos os arquivos", "*.*"))
        )
        if path:
            self.custom_model_path_var.set(path)
            logger.info(f"Modelo personalizado selecionado: {path}")

    def on_model_selection_change(self, choice): # Para ComboBox pré-definido
        logger.info(f"Modelo pré-definido selecionado: {choice}")
        # Não precisa recarregar imediatamente, espera por "Aplicar"

    def on_gpu_toggle(self):
        logger.info(f"Opção GPU alterada para: {self.gpu_var.get()}")
        # Não precisa recarregar imediatamente, espera por "Aplicar"

    def apply_model_changes(self):
        """Aplica as mudanças de modelo (caminho e dispositivo) e recarrega."""
        new_model_path = ""
        if self.model_source_var.get() == "predefined":
            model_key = self.predefined_model_var.get()
            new_model_path = PREDEFINED_MODELS.get(model_key, "yolov8n.pt")
        else: # custom
            new_model_path = self.custom_model_path_var.get()
            if not new_model_path or not Path(new_model_path).exists():
                self.show_error("Caminho do modelo personalizado inválido ou não encontrado.")
                return

        new_device = "cuda" if self.gpu_var.get() and torch.cuda.is_available() else "cpu"

        if self.detector and new_model_path == self.current_model_path and new_device == self.current_device:
            self.update_status("Nenhuma alteração no modelo ou dispositivo.")
            return

        self.load_yolo_model(new_model_path, new_device)

    def _load_model_worker(self, model_path: str, device: str, initial_load: bool = False):
        """Função worker para carregar o modelo em uma thread separada."""
        self.window.after(0, lambda: self.toggle_button.configure(state="disabled"))
        self.window.after(0, lambda: self.apply_model_button.configure(state="disabled", text="Carregando Modelo..."))
        self.window.after(0, lambda: self.update_status(f"Carregando modelo {Path(model_path).name} em {device}..."))

        try:
            if self.detector and self.running: # Se estiver rodando, para a detecção primeiro
                self.window.after(0, self.stop_detection) # Agenda stop_detection na thread principal
                if self.detection_thread and self.detection_thread.is_alive():
                    self.detection_thread.join(timeout=2.0) # Espera parar

            new_detector = ObjectDetector(
                model_path=model_path,
                device=device,
                conf_threshold=self.confidence_var.get(),
                iou_threshold=self.iou_var.get(),
                max_det=self.max_det_var.get()
            )
            self.detector = new_detector
            self.current_model_path = model_path
            self.current_device = device
            self.window.after(0, lambda: self.update_status(f"Modelo {Path(model_path).name} carregado com sucesso em {device}."))
            logger.info(f"Modelo {Path(model_path).name} carregado em {device}.")

        except Exception as e:
            self.detector = None # Garante que detector é None se o carregamento falhar
            self.window.after(0, lambda: self.show_error(f"Falha ao carregar modelo: {e}"))
            logger.error(f"Falha ao carregar modelo: {e}", exc_info=True)
        finally:
            self.window.after(0, lambda: self.toggle_button.configure(state="normal"))
            self.window.after(0, lambda: self.apply_model_button.configure(state="normal", text="Aplicar e Recarregar Modelo"))
            # Se foi um carregamento inicial e falhou, o app pode ficar inutilizável.
            # Se estava rodando e falhou, a detecção deve permanecer parada.
            if self.detector is None and not initial_load and self.running:
                self.window.after(0, self.stop_detection) # Garante que está parado

    def load_yolo_model(self, model_path: str, device: str, initial_load: bool = False):
        """Inicia o carregamento do modelo YOLO em uma nova thread."""
        if self.model_load_thread and self.model_load_thread.is_alive():
            self.update_status("Carregamento de modelo já em progresso.")
            return

        logger.info(f"Iniciando carregamento do modelo: {model_path} para {device}")
        self.model_load_thread = Thread(target=self._load_model_worker, args=(model_path, device, initial_load), daemon=True)
        self.model_load_thread.start()

    def _create_detection_tab(self, tab):
        """Cria os controles da aba 'Detecção'."""
        ctk.CTkLabel(tab, text="Parâmetros de Detecção", font=ctk.CTkFont(weight="bold")).pack(pady=10, anchor="w", padx=10)

        # Confiança
        ctk.CTkLabel(tab, text="Limiar de Confiança:").pack(anchor="w", padx=10, pady=(5,0))
        self.confidence_var = ctk.DoubleVar()
        self.confidence_slider = ctk.CTkSlider(tab, from_=0.01, to=1.0, number_of_steps=99, variable=self.confidence_var, command=self.update_confidence_label)
        self.confidence_slider.pack(fill="x", padx=10, pady=(0,0))
        self.conf_value_label = ctk.CTkLabel(tab, text="")
        self.conf_value_label.pack(anchor="e", padx=10, pady=(0,10))

        # IOU
        ctk.CTkLabel(tab, text="Limiar de IOU (Sobreposição):").pack(anchor="w", padx=10, pady=(5,0))
        self.iou_var = ctk.DoubleVar()
        self.iou_slider = ctk.CTkSlider(tab, from_=0.01, to=1.0, number_of_steps=99, variable=self.iou_var, command=self.update_iou_label)
        self.iou_slider.pack(fill="x", padx=10, pady=(0,0))
        self.iou_value_label = ctk.CTkLabel(tab, text="")
        self.iou_value_label.pack(anchor="e", padx=10, pady=(0,10))

        # Máximo de Detecções
        ctk.CTkLabel(tab, text="Máximo de Detecções:").pack(anchor="w", padx=10, pady=(5,0))
        self.max_det_var = ctk.IntVar()
        self.max_det_slider = ctk.CTkSlider(tab, from_=1, to=500, number_of_steps=499, variable=self.max_det_var, command=self.update_max_det_label)
        self.max_det_slider.pack(fill="x", padx=10, pady=(0,0))
        self.max_det_value_label = ctk.CTkLabel(tab, text="")
        self.max_det_value_label.pack(anchor="e", padx=10, pady=(0,10))

    def update_confidence_label(self, value=None):
        """Atualiza o label de confiança e o parâmetro no detector."""
        val = self.confidence_var.get()
        self.conf_value_label.configure(text=f"{val:.2f}")
        if self.detector: self.detector.update_parameters(conf_threshold=val)

    def update_iou_label(self, value=None):
        """Atualiza o label de IOU e o parâmetro no detector."""
        val = self.iou_var.get()
        self.iou_value_label.configure(text=f"{val:.2f}")
        if self.detector: self.detector.update_parameters(iou_threshold=val)

    def update_max_det_label(self, value=None):
        """Atualiza o label de máximo de detecções e o parâmetro no detector."""
        val = self.max_det_var.get()
        self.max_det_value_label.configure(text=f"{val}")
        if self.detector: self.detector.update_parameters(max_det=val)

    def _create_save_tab(self, tab):
        """Cria os controles da aba 'Salvar'."""
        ctk.CTkLabel(tab, text="Gerenciar Configurações", font=ctk.CTkFont(weight="bold")).pack(pady=10, anchor="w", padx=10)
        save_button = ctk.CTkButton(tab, text="Salvar Configurações Atuais", command=self.save_config)
        save_button.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(tab, text="As configurações serão salvas em 'config_aprimorado.yaml'.", wraplength=tab.winfo_width()-20).pack(pady=5, padx=10)


    def setup_shortcuts(self):
        """Configura os atalhos de teclado."""
        self.pressed_keys = set()
        def on_press(key):
            if not self.window or not self.window.winfo_exists(): # Verifica se a janela existe
                 if self.keyboard_listener: self.keyboard_listener.stop(); return
            try:
                target_key = getattr(key, 'char', key) # Usa char se for letra/num, senão o objeto Key
                if target_key is not None and target_key not in self.pressed_keys:
                    self.pressed_keys.add(target_key)
                    if target_key == ' ': self.window.after(0, self.toggle_detection)
                    elif key == keyboard.Key.esc: self.window.after(0, self.on_closing)
            except Exception as e: logger.warning(f"Erro ao processar tecla pressionada: {e}")

        def on_release(key):
             if not self.window or not self.window.winfo_exists(): return
             try:
                 target_key = getattr(key, 'char', key)
                 if target_key in self.pressed_keys: self.pressed_keys.remove(target_key)
             except Exception as e: logger.warning(f"Erro ao processar tecla liberada: {e}")
        try:
            self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release, suppress=False)
            self.keyboard_listener.start()
            logger.info("Atalhos de teclado ativados (Espaço: Iniciar/Parar, Esc: Sair)")
        except Exception as e: # Frequentemente devido a problemas com o servidor de display no Linux
            logger.error(f"Falha ao iniciar listener de teclado: {e}. Atalhos podem não funcionar.")
            self.keyboard_listener = None


    def toggle_detection(self):
        """Inicia ou para a detecção."""
        if not self.detector:
            self.show_error("Modelo não carregado. Por favor, configure e aplique um modelo na aba 'Modelo'.")
            self.update_status("Falha ao iniciar: Modelo não carregado.")
            return

        if not self.running:
            # --- Iniciar Detecção ---
            self.toggle_button.configure(state="disabled", text="Iniciando...")
            self.update_status("Tentando iniciar fonte de vídeo...")
            self.window.update_idletasks()

            self.is_video_file_mode = self.input_source_var.get() == "video_file"

            try:
                if self.is_video_file_mode:
                    if not self.video_file_path or not Path(self.video_file_path).exists():
                        raise IOError("Arquivo de vídeo não selecionado ou não encontrado.")
                    self.video_capture = cv2.VideoCapture(self.video_file_path)
                    if not self.video_capture.isOpened():
                        raise IOError(f"Não foi possível abrir o arquivo de vídeo: {self.video_file_path}")
                    source_name = Path(self.video_file_path).name
                    logger.info(f"Vídeo {source_name} aberto com sucesso.")
                else: # Modo Câmera
                    cam_index = self.camera_index_var.get()
                    # Tenta backend V4L2 no Linux primeiro, depois CAP_ANY como fallback ou para outros OS
                    backend_to_try = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY
                    self.camera = cv2.VideoCapture(cam_index, backend_to_try)
                    if not self.camera or not self.camera.isOpened():
                        logger.warning(f"Falha ao abrir câmera {cam_index} com backend V4L2/Padrão. Tentando CAP_ANY...")
                        self.camera = cv2.VideoCapture(cam_index, cv2.CAP_ANY) # Fallback
                        if not self.camera or not self.camera.isOpened():
                            raise IOError(f"Não foi possível abrir câmera índice {cam_index}.")

                    # Define propriedades da câmera (melhor esforço)
                    cam_w, cam_h, cam_fps = self.config['camera_width'], self.config['camera_height'], self.config['camera_fps']
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
                    self.camera.set(cv2.CAP_PROP_FPS, cam_fps) # Pode ser ignorado por algumas câmeras/drivers
                    aw = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ah = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    afps = self.camera.get(cv2.CAP_PROP_FPS)
                    source_name = f"Câmera {cam_index}"
                    logger.info(f"Câmera {cam_index} aberta. Solicitado: {cam_w}x{cam_h}@{cam_fps}fps. Real: {aw}x{ah}@{afps if afps > 0 else 'N/A'}fps")

                # Testa leitura de um frame
                cap = self.video_capture if self.is_video_file_mode else self.camera
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    # Libera o recurso se a leitura falhar
                    if self.is_video_file_mode and self.video_capture: self.video_capture.release(); self.video_capture = None
                    elif not self.is_video_file_mode and self.camera: self.camera.release(); self.camera = None
                    raise IOError(f"Falha ao ler o primeiro frame de {source_name}.")

                # Se for arquivo de vídeo, retorna o frame para o início
                if self.is_video_file_mode and self.video_capture:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)


                self.running = True
                self.toggle_button.configure(text="Parar Detecção", state="normal")
                if self.detection_thread and self.detection_thread.is_alive(): # Junta thread anterior se houver
                    self.detection_thread.join(timeout=1.0)
                self.detection_thread = Thread(target=self.detection_loop, daemon=True)
                self.detection_thread.start()
                self.update_status(f"Detecção em execução em {source_name}...")
                logger.info(f"Detecção iniciada em {source_name}.")

            except Exception as e:
                self.running = False # Garante que running é falso
                logger.error(f"Falha ao iniciar detecção/fonte: {e}\n{traceback.format_exc()}")
                self.show_error(f"Erro ao iniciar: {e}")
                if self.camera: self.camera.release(); self.camera = None
                if self.video_capture: self.video_capture.release(); self.video_capture = None
                self.toggle_button.configure(text="Iniciar Detecção", state="normal")
                self.update_status("Erro ao iniciar detecção.")
        else:
            # --- Parar Detecção ---
            self.stop_detection()

    def stop_detection(self):
        """Para a thread de detecção e libera a câmera/vídeo."""
        if not self.running and not (self.detection_thread and self.detection_thread.is_alive()):
            logger.info("Parar detecção chamado, mas já estava parada ou parando.")
            # Garante que a UI está no estado parado se, por algum motivo, não estiver
            if hasattr(self, 'window') and self.window.winfo_exists():
                self.window.after(0, self.stop_detection_ui_update)
            return

        logger.info("Parando processo de detecção...")
        self.running = False # Sinaliza para a thread parar
        if hasattr(self, 'toggle_button') and self.toggle_button.winfo_exists():
            self.toggle_button.configure(text="Parando...", state="disabled")
        if hasattr(self, 'window') and self.window.winfo_exists():
            self.window.update_idletasks()

        thread_to_join = self.detection_thread
        self.detection_thread = None # Define como None antes de join para evitar race conditions
        if thread_to_join and thread_to_join.is_alive():
            logger.info("Aguardando thread de detecção terminar...")
            thread_to_join.join(timeout=3.0) # Timeout um pouco maior
            if thread_to_join.is_alive(): logger.warning("Thread de detecção não parou graciosamente.")
            else: logger.info("Thread de detecção finalizada.")
        else: logger.info("Thread de detecção não estava rodando ou já finalizada.")

        # Libera captura de câmera/vídeo
        if self.camera:
            try: self.camera.release(); logger.info("Câmera liberada.")
            except Exception as e: logger.error(f"Exceção ao liberar câmera: {e}")
            self.camera = None
        if self.video_capture:
            try: self.video_capture.release(); logger.info("Captura de vídeo liberada.")
            except Exception as e: logger.error(f"Exceção ao liberar captura de vídeo: {e}")
            self.video_capture = None

        # Atualiza UI no estado parado (agenda na thread principal)
        if hasattr(self, 'window') and self.window.winfo_exists():
             self.window.after(0, self.stop_detection_ui_update)
        else: # Se a janela sumiu, chama diretamente para log/limpeza
             self.stop_detection_ui_update()
        logger.info("Detecção parada.")


    def detection_loop(self):
        """Loop principal para captura de frames e detecção."""
        logger.info("Loop de detecção iniciado.")
        cap = self.video_capture if self.is_video_file_mode else self.camera

        while self.running:
            if cap is None or not cap.isOpened():
                logger.error("Fonte de vídeo não disponível ou fechada inesperadamente.")
                self.running = False # Garante término do loop
                if hasattr(self, 'window') and self.window.winfo_exists():
                    self.window.after(0, lambda: self.show_error("Conexão com a fonte de vídeo perdida."))
                    self.window.after(0, self.stop_detection_ui_update) # Garante atualização da UI
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                if self.is_video_file_mode: # Se for arquivo de vídeo
                    if self.video_loop_var.get(): # Se loop estiver ativo
                        logger.info("Fim do vídeo, reiniciando (loop).")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reinicia o vídeo
                        continue
                    else: # Fim do vídeo sem loop
                        logger.info("Fim do vídeo. Parando detecção.")
                        self.running = False # Sinaliza para parar
                        if hasattr(self, 'window') and self.window.winfo_exists():
                           self.window.after(0, self.stop_detection) # Chama stop_detection completo
                        break
                else: # Falha no stream da câmera
                    logger.warning("Falha ao capturar frame da câmera. Tentando novamente...")
                    time.sleep(0.1) # Pequena pausa antes de tentar novamente
                    continue

            # FPS calculation moved to update_interface to be based on displayed frames

            if self.detector:
                processed_frame, detections = self.detector.process_frame(frame.copy()) # Passa uma cópia
                if hasattr(self, 'window') and self.window.winfo_exists():
                    self.window.after(0, lambda pf=processed_frame, d=detections: self.update_interface(pf, d))
            else:
                # Desenha no frame não processado se o detector não estiver pronto
                if hasattr(self, 'window') and self.window.winfo_exists():
                    self.window.after(0, lambda pf=frame: self.update_interface(pf, []))
                time.sleep(0.03) # Espera aproximada de 30fps se não houver processamento

            # Pequena pausa para ceder controle se o processamento for muito rápido
            # time.sleep(0.001)

        logger.info("Loop de detecção finalizado.")
        # Garante que a UI atualize corretamente quando o loop sair, independentemente do motivo
        if not self.running and hasattr(self, 'window') and self.window.winfo_exists():
            self.window.after(0, self.stop_detection_ui_update)


    def stop_detection_ui_update(self):
        """Atualiza a UI para o estado parado."""
        logger.debug("Atualizando UI para estado parado.")
        if hasattr(self, 'toggle_button') and self.toggle_button.winfo_exists():
            self.toggle_button.configure(text="Iniciar Detecção", state="normal")
        if hasattr(self, 'fps_label') and self.fps_label.winfo_exists():
            self.fps_label.configure(text="FPS: 0")
        if hasattr(self, 'video_label') and self.video_label.winfo_exists():
            # Limpa o label de vídeo - mostra uma imagem/texto placeholder
            # Usa display_width da config para o placeholder, mantendo uma proporção comum (ex: 16:9)
            ph_w = self.config.get('display_width', 800)
            ph_h = int(ph_w * 9/16) # Proporção 16:9
            placeholder_img = Image.new("RGB", (ph_w, ph_h), color="black")
            ctk_placeholder = ctk.CTkImage(placeholder_img, size=(placeholder_img.width, placeholder_img.height))
            self.video_label.configure(image=ctk_placeholder, text="Feed parado. Clique em Iniciar.") # Mantém placeholder
            self.video_label.image = ctk_placeholder # Importante para manter referência
        if hasattr(self, 'objects_label') and self.objects_label.winfo_exists():
            self.objects_label.configure(text="Objetos: 0")
        self.update_status("Detecção parada")

    def update_interface(self, frame: np.ndarray, detections: List[Dict] = None):
        """Atualiza o label de vídeo com o frame processado e informações."""
        if frame is None: return
        # Verifica se a UI está pronta ou se desapareceu
        if not (hasattr(self, 'window') and self.window.winfo_exists() and hasattr(self, 'video_label') and self.video_label.winfo_exists()):
            return

        try:
            # Redimensiona o frame para exibição
            # Usa o tamanho atual do video_label para melhor adaptação
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            if label_width <=1 or label_height <=1: # Ainda não desenhado ou muito pequeno
                # Fallback para display_width da config, mantém proporção do frame original
                target_w = self.config.get('display_width', 800)
                target_h = int(target_w * (frame.shape[0] / frame.shape[1]) if frame.shape[1] > 0 else target_w * 9/16)
            else:
                # Escala para caber no video_label mantendo a proporção
                frame_aspect = frame.shape[1] / frame.shape[0] # largura / altura
                label_aspect = label_width / label_height

                if frame_aspect > label_aspect: # Frame é mais largo que a área do label
                    target_w = label_width
                    target_h = int(target_w / frame_aspect)
                else: # Frame é mais alto ou mesma proporção
                    target_h = label_height
                    target_w = int(target_h * frame_aspect)

            if target_w <=0 or target_h <=0: # Verificação de segurança para dimensões
                logger.warning(f"Dimensões alvo inválidas para exibição do frame: {target_w}x{target_h}. Usando 320x240.")
                target_w, target_h = 320, 240 # Fallback mínimo

            frame_resized = cv2.resize(frame, (target_w, target_h))

            # Cálculo de FPS baseado nos frames realmente mostrados
            now = time.time()
            self.frame_count += 1
            elapsed = now - self.last_fps_update
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                if hasattr(self, 'fps_label') and self.fps_label.winfo_exists():
                    self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
                self.frame_count = 0
                self.last_fps_update = now

            cv2.putText(frame_resized, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if detections is not None and hasattr(self, 'objects_label') and self.objects_label.winfo_exists():
                self.objects_label.configure(text=f"Objetos: {len(detections)}")
                # Opcionalmente, desenhe um resumo das detecções no frame_resized aqui, se desejar
                # y_offset = 50
                # for i, det in enumerate(detections[:3]): # Mostra os 3 primeiros
                #     text = f"{det['label']} ({det['confidence']:.2f})"
                #     cv2.putText(frame_resized, text, (10, y_offset + i*20),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,200,50), 1, cv2.LINE_AA)

            # Converte para formato PIL e depois CTkImage
            image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ctk.CTkImage(image, size=(target_w, target_h))

            # Atualiza o label
            self.video_label.configure(image=photo, text="") # Limpa qualquer texto de "aguardando"
            self.video_label.image = photo # Mantém referência
        except Exception as e:
            logger.error(f"Erro ao atualizar interface: {e}\n{traceback.format_exc()}")

    def update_status(self, message: str):
        """Atualiza a barra de status com uma mensagem e timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        logger.info(f"Status: {message}")
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            if hasattr(self, 'window') and self.window.winfo_exists():
                self.window.after(0, lambda msg=full_message: self.status_label.configure(text=msg))

    def show_error(self, message: str):
        """Mostra uma messagebox de erro e atualiza o status."""
        logger.error(message)
        if hasattr(self, 'window') and self.window.winfo_exists():
             self.window.after(0, lambda msg=message: messagebox.showerror("Erro", msg, parent=self.window))
        self.update_status(f"Erro: {message[:100]}...") # Mostra apenas parte da msg no status

    def on_closing(self):
        """Chamado quando a janela é fechada."""
        logger.info("Fechamento da janela solicitado. Iniciando limpeza...")
        if self.running:
            user_choice = messagebox.askyesnocancel("Sair?", "A detecção está em execução. Deseja parar e sair?", parent=self.window)
            if user_choice is True: # Sim
                self.cleanup()
            elif user_choice is False: # Não (não fechar)
                return
            else: # Cancelar
                return
        else:
            self.cleanup()


    def cleanup(self):
        """Libera todos os recursos."""
        if self._cleanup_called:
            logger.warning("Limpeza já chamada ou em progresso.")
            return
        self._cleanup_called = True
        logger.info("--- Iniciando Limpeza da Aplicação ---")

        self.running = False # Garante que todos os loops dependentes disso parem

        # Para thread de detecção e libera câmera/vídeo
        if self.detection_thread and self.detection_thread.is_alive():
            logger.info("Limpeza: Aguardando thread de detecção...")
            self.detection_thread.join(timeout=2.0)
        self.detection_thread = None

        # Para thread de carregamento de modelo
        if self.model_load_thread and self.model_load_thread.is_alive():
            logger.info("Limpeza: Aguardando thread de carregamento de modelo...")
            self.model_load_thread.join(timeout=1.0) # Pode estar presa em um carregamento longo
        self.model_load_thread = None

        if self.camera:
            try: self.camera.release()
            except Exception as e: logger.error(f"Erro ao liberar câmera na limpeza: {e}")
            self.camera = None
        if self.video_capture:
            try: self.video_capture.release()
            except Exception as e: logger.error(f"Erro ao liberar vídeo na limpeza: {e}")
            self.video_capture = None

        if self.keyboard_listener:
            logger.info("Limpeza: Parando listener de teclado...")
            try:
                self.keyboard_listener.stop()
            except Exception as e: logger.error(f"Erro ao parar listener de teclado: {e}")
            self.keyboard_listener = None

        if self.window:
            logger.info("Limpeza: Destruindo janela...")
            try:
                # Cancela chamadas after() pendentes, se possível (de forma robusta)
                if hasattr(self.window, 'tk'):
                    for after_id in self.window.tk.eval('after info').split():
                        try: self.window.after_cancel(after_id)
                        except: pass # Ignora erros se ID for inválido
                self.window.destroy()
                logger.info("Janela destruída.")
            except Exception as e: logger.error(f"Erro ao destruir janela: {e}")
            self.window = None

        logger.info("--- Limpeza da Aplicação Concluída ---")
        # Força saída se cleanup for chamado de um erro crítico no início da inicialização
        if not hasattr(self.window, 'mainloop') or (self.window and not self.window.winfo_exists()):
            sys.exit(0) # Sai se o mainloop não estiver rodando / janela sumiu

# --- Execução Principal ---
def main():
    app = None
    # Configuração do handler global de exceções
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.critical("Exceção não tratada capturada!", exc_info=(exc_type, exc_value, exc_traceback))
        # Constrói uma mensagem amigável
        error_message = f"Ocorreu um erro crítico inesperado:\n{exc_value}\n\n" \
                        f"A aplicação será fechada.\n" \
                        f"Por favor, verifique o arquivo 'navigation_aprimorado.log' para detalhes."

        # Tenta mostrar uma messagebox, mesmo que CustomTkinter possa estar envolvido
        try:
            # Usa tkinter básico para esta mensagem de erro crítica para evitar problemas com CTk
            import tkinter as tk_basic
            from tkinter import messagebox as tk_messagebox_basic
            root_err = tk_basic.Tk()
            root_err.withdraw() # Esconde a janela principal do Tk
            tk_messagebox_basic.showerror("Erro Crítico", error_message, parent=None) # Sem pai
            root_err.destroy()
        except Exception as e_msg:
            print(f"ERRO CRÍTICO (falha ao mostrar messagebox): {error_message}\nDetalhes da falha do messagebox: {e_msg}", file=sys.stderr)

        if app and not getattr(app, '_cleanup_called', False):
            app.cleanup() # Tenta limpeza
        sys.exit(1) # Sai com código de erro

    sys.excepthook = handle_exception

    try:
        logger.info("================================================")
        logger.info(" Iniciando Aplicação de Navegação (Aprimorado)...")
        logger.info("================================================")

        app = NavigationApp()

        if app.window: # Verifica se a janela foi criada com sucesso
            logger.info("Entrando no loop de eventos principal (mainloop)...")
            app.window.mainloop()
            logger.info("Loop principal da aplicação finalizado normalmente.")
        else:
            logger.error("Inicialização da UI principal falhou. Não é possível iniciar mainloop.")
            if app and not getattr(app, '_cleanup_called', False): app.cleanup()
            sys.exit(1)

    except SystemExit: # Permite que sys.exit() passe
        logger.info("Aplicação terminando via SystemExit.")
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt recebido. Limpando e saindo...")
        if app and not getattr(app, '_cleanup_called', False): app.cleanup()
        sys.exit(0)
    except Exception as e: # Captura qualquer outra exceção durante a configuração antes do mainloop
        logger.critical(f"Erro fatal durante a inicialização antes do mainloop: {e}\n{traceback.format_exc()}")
        # O sys.excepthook idealmente deveria pegar isso, mas como fallback:
        error_message_init = f"Falha crítica na inicialização: {e}\nVerifique 'navigation_aprimorado.log'."
        try:
            import tkinter as tk_basic; from tkinter import messagebox as tk_messagebox_basic
            root = tk_basic.Tk(); root.withdraw()
            tk_messagebox_basic.showerror("Erro Fatal de Inicialização", error_message_init)
            root.destroy()
        except Exception as me_init:
            print(f"ERRO FATAL DE INICIALIZAÇÃO (sem GUI): {error_message_init}\nDetalhes do messagebox: {me_init}", file=sys.stderr)

        if app and not getattr(app, '_cleanup_called', False): app.cleanup()
        sys.exit(1)
    finally:
        logger.info("Ponto final de saída da aplicação atingido.")
        if app and not getattr(app, '_cleanup_called', False) : # Garante limpeza se ainda não foi feita
             logger.info("Executando verificação final de limpeza no finally...")
             app.cleanup()
        logger.info("================================================")
        logging.shutdown() # Garante que todos os logs sejam escritos

if __name__ == "__main__":
    # Correção para PyInstaller/cx_Freeze ao usar multiprocessing ou pynput em algumas plataformas
    # from multiprocessing import freeze_support
    # freeze_support() # Descomente se for empacotar com PyInstaller e usar recursos de multiprocessing

    # Consciência de DPI alto para Windows (melhora renderização de fontes)
    if sys.platform == "win32":
        try:
            from ctypes import windll
            # Argumento 1 para Per Monitor DPI Awareness
            # Argumento 2 para System DPI Awareness
            windll.shcore.SetProcessDpiAwareness(1) 
        except Exception as e:
            logger.warning(f"Não foi possível definir consciência de DPI: {e}")
    main()