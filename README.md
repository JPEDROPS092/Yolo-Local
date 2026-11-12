# ObstacleSense AI

Núcleo em Python do **ObstacleSense AI**, projeto para a SEMINFO 2026 (IFAM). A pessoa aponta a câmera do celular ou do computador para o caminho e o sistema avisa, em português e em voz alta, o que está à frente e para que lado desviar.

O programa detecta obstáculos com YOLO11 (o YOLOv8 continua disponível), estima a distância em metros, classifica a direção e o perigo, e entrega a mesma leitura para a tela, para a voz e para um aplicativo Android.

## O que mudou em relação ao detector local

O `config.yaml` já tinha `audio_interval`, `audio_rate`, `proximity_warning_distance` e `collision_warning_threshold`, mas o programa não lia esse arquivo e só dizia se o objeto parecia perto, médio ou longe. Agora esses valores entram na fala, no cálculo de risco e no serviço HTTP.

## Arquitetura

1. **Detector** (`obstaclesense/detector.py`): YOLO11 nano por padrão, ou qualquer peso YOLO (`yolov8n.pt`, um `.pt` próprio, um `.onnx`). Só entram as classes de obstáculo listadas no config.
2. **Distância** (`obstaclesense/distance.py`): estimativa geométrica sempre ligada, pelo tamanho real típico da classe e pelo campo de visão. O contato com o chão entra quando o pé do objeto está visível.
3. **Profundidade opcional** (`obstaclesense/depth.py`): Depth Anything V2 Small. A escala relativa é calibrada pelas distâncias geométricas do mesmo quadro e as duas leituras são combinadas.
4. **Direção e perigo** (`direction.py`, `danger.py`): esquerda, centro ou direita do caminho. Veículo, pouca distância e objeto no centro sobem a nota. `collision_warning_threshold` é o corte da nota alta.
5. **Rastreamento** (`tracking.py`): associa a mesma caixa entre quadros. Se a distância cai, a frase diz que o objeto está se aproximando.
6. **Fala** (`phrases.py`, `speech.py`): uma frase em português do Brasil, no máximo a cada `audio_interval` segundos, sobre o obstáculo mais perigoso. Um alerta alto pode interromper o intervalo. A voz é offline: Piper se houver um modelo pt-BR, senão espeak-ng com a voz `pt-br`, senão pyttsx3.
7. **Saídas**: janela CustomTkinter, terminal com OpenCV, e API FastAPI para o Android.

## Requisitos

- Python 3.10 ou mais novo
- CPU basta. GPU CUDA é usada só se `enable_gpu` estiver ligado e o PyTorch enxergar a placa
- `espeak-ng` para a voz em português

## Instalação

No Linux, a voz pt-BR vem do espeak-ng:

```bash
sudo apt install espeak-ng
```

O PyTorch de CPU evita o pacote grande com CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

O primeiro uso baixa o `yolo11n.pt` (alguns megabytes).

Profundidade monocular, se quiser ligar:

```bash
pip install -r requirements-depth.txt
```

## Configuração

Tudo fica em `config.yaml`.

| Chave | Papel |
| --- | --- |
| `yolo_model_path` | `yolo11n.pt` ou `yolov8n.pt` |
| `imgsz` | tamanho da inferência; 320 é o padrão leve |
| `obstacle_classes` | classes que importam para quem está caminhando |
| `audio_interval` | segundos mínimos entre uma fala e outra |
| `audio_rate` | velocidade da voz |
| `proximity_warning_distance` | distância, em metros, tratada como perto |
| `collision_warning_threshold` | nota a partir da qual o perigo é alto |
| `minimum_object_size` | caixa muito pequena some da lista (ruído distante) |
| `hfov_degrees`, `camera_height_m` | geometria do celular na altura do peito |
| `depth_enabled` | liga o Depth Anything V2 Small |
| `tts_engine` | `auto`, `pyttsx3`, `espeak` ou `piper` |
| `piper_model_path` | arquivo `.onnx` de uma voz Piper pt_BR |

Para o Piper, baixe uma voz brasileira (por exemplo `pt_BR-faber-medium`) e aponte `piper_model_path` para o `.onnx`. Com `tts_engine: piper` e o executável `piper` no PATH, essa voz é usada.

## Demonstração

Imagem de exemplo, sem abrir janela. O JSON e a frase aparecem no terminal, e a foto anotada fica em `outputs/demo.jpg`:

```bash
python main.py --demo --no-display
```

Câmera:

```bash
python main.py --cli --source 0
```

Arquivo de vídeo, gravando o resultado:

```bash
python main.py --cli --source caminho/video.mp4 --output outputs/resultado.mp4
```

YOLOv8 em vez do YOLO11:

```bash
python main.py --model yolov8n.pt --demo --no-display
```

Interface gráfica (câmera ou vídeo, confiança, troca de modelo, voz):

```bash
python main.py
```

Atalhos: espaço inicia ou para, Esc fecha. A caixa fica vermelha no perigo alto, laranja no médio e verde no baixo, com distância e direção em português. O FPS e a latência aparecem no quadro.

Sem alto-falante, a mesma frase pode ser gravada:

```bash
espeak-ng -v pt-br -s 175 -w outputs/aviso.wav "Existe uma pessoa aproximadamente dois metros à sua frente. Desvie para a direita."
```

O programa faz o equivalente por `obstaclesense.speech.synthesize_to_file`.

## API para o Android

```bash
python main.py --api
```

O serviço escuta em `0.0.0.0:8000`. No celular, use o IP da máquina.

`POST /analyze` recebe a imagem no campo `file` (multipart) e responde:

```json
{
  "guidance": "Existe uma pessoa aproximadamente dois metros à sua frente. Desvie para a direita.",
  "latency_ms": 80.0,
  "fps": 12.0,
  "objects": [
    {
      "label": "person",
      "confidence": 0.88,
      "distance_meters": 2.1,
      "direction": "center",
      "danger_level": "high"
    }
  ]
}
```

Os objetos usam exatamente estes campos: `label`, `confidence`, `distance_meters`, `direction` (`left`, `center`, `right`) e `danger_level` (`low`, `medium`, `high`). A frase em `guidance` é só em português. `latency_ms` e `fps` vêm junto para acompanhar o custo de cada quadro.

```bash
curl -F "file=@outputs/bus.jpg" http://127.0.0.1:8000/analyze
```

`GET /health` devolve o modelo, o dispositivo e as classes.

WebSocket em `/ws`: o cliente envia os bytes JPEG de cada quadro (ou um texto JSON `{"image_base64": "..."}`) e recebe o mesmo objeto da rota `/analyze`. A voz fica no aplicativo; a API não fala.

## Exportar para o aparelho

```bash
pip install -r requirements-export.txt
python main.py --export onnx
python main.py --export tflite
```

O ONNX abre de novo neste programa se `yolo_model_path` apontar para o arquivo gerado. O TFLite pede TensorFlow instalado e serve para o Android rodar o detector sem este serviço.

## Testes

A lógica de distância, direção, perigo, rastreamento e frase não precisa do modelo:

```bash
pytest
```

## Latência

O padrão é de propósito pequeno: YOLO11n, imagem de entrada até 640×480, inferência em 320 px, profundidade desligada e no máximo 30 caixas. Cada resultado imprime FPS e milissegundos. Subir `imgsz` ou ligar `--depth` melhora a leitura e custa tempo de CPU. A profundidade roda a cada `depth_every_n_frames` quadros e, se falhar, a estimativa geométrica continua.

Na verificação em CPU, com o modelo já carregado, a foto de exemplo ficou em cerca de 12 ms por quadro na análise e cerca de 25 ms com as caixas desenhadas. Um celular mandando foto grande é reduzido até `frame_width` × `frame_height` sem esticar a imagem.

## Aviso

A distância é uma estimativa. Ela depende do campo de visão e da altura da câmera, e o tamanho real de cada classe é um valor típico, não uma medida do objeto. O sistema ajuda a desviar. Não substitui a bengala, o cão-guia nem a atenção de quem caminha.
