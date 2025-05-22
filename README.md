# Sistema de Detecção de Objetos com YOLO

Sistema de detecção de objetos em tempo real usando YOLOv8, com suporte para câmera e vídeos.

## Características

- Interface gráfica moderna e intuitiva
- Suporte para câmera e arquivos de vídeo
- Múltiplos modelos YOLOv8 disponíveis:
  - YOLOv8n (nano - mais rápido)
  - YOLOv8s (small)
  - YOLOv8m (medium)
  - YOLOv8l (large)
  - YOLOv8x (extra large - mais preciso)
- Carregamento de modelos personalizados
- Configurações ajustáveis em tempo real
- Suporte para GPU (CUDA) e CPU

## Requisitos

- Python 3.8+
- OpenCV
- PyYAML
- NumPy
- CustomTkinter
- Ultralytics (YOLOv8)
- Pillow
- Pynput

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso

1. Execute o programa:

```bash
python main.py
```

2. Interface:

   - **Aba Principal**: Controles básicos e informações
   - **Aba Câmera**: Configurações da câmera/vídeo
   - **Aba Modelo**: Seleção e configuração do modelo YOLO
   - **Aba Detecção**: Parâmetros de detecção
   - **Aba Segurança**: Configurações de alertas
   - **Aba Visual**: Personalização da interface
3. Entrada de Vídeo:

   - Câmera: Selecione o índice da câmera (geralmente 0 para webcam)
   - Arquivo: Clique em "Carregar Vídeo" e selecione um arquivo
4. Modelos:

   - Use os modelos pré-treinados ou
   - Carregue seu próprio modelo treinado (.pt)

## Configurações

Todas as configurações são salvas em `config.yaml`:

```yaml
# Configurações da câmera
camera_index: 0
frame_width: 1280
frame_height: 720
fps_target: 30

# Configurações do modelo
yolo_model_path: "yolov8n.pt"
enable_gpu: true

# Configurações de detecção
confidence_threshold: 0.5
iou_threshold: 0.45
max_det: 50
```

## Atalhos de Teclado

- `Espaço`: Iniciar/Parar detecção
- `Esc`: Sair do programa

## Contribuindo

1. Fork o projeto
2. Crie sua branch de feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
