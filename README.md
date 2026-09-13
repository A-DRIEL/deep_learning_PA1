# Deep Learning PA1 — Segmentação de Instâncias

Segmentação de instâncias de núcleos DSB2018, usando arquiteturas
de segmentação semântica adaptadas para produzir rótulos instance-aware
## Ambiente

Este projeto usa [uv](https://docs.astral.sh/uv/) para gerenciar dependências.

```powershell
# instalar uv, se necessário
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# sincronizar dependências -- escolha conforme seu hardware
uv sync --extra gpu    # com GPU NVIDIA (CUDA 12.8+)
uv sync --extra cpu    # sem GPU
```

Confirmar que funcionou:
```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Download dos dados

Dataset: DSB2018 (BBBC038v1), via Kaggle API.

```powershell
uv run kaggle auth login
# se for a primeira vez, aceite as regras da competição em:
# https://www.kaggle.com/competitions/data-science-bowl-2018/rules

New-Item -ItemType Directory -Force -Path data\raw
uv run kaggle competitions download -c data-science-bowl-2018 -f stage1_train.zip -p data\raw
Expand-Archive -Path "data\raw\stage1_train.zip" -DestinationPath "data\raw\stage1_train" -Force
Remove-Item data\raw\stage1_train.zip
```

Confirme que baixou corretamente (deve mostrar 670):
```powershell
uv run python -c "from pathlib import Path; print(len(list(Path('data/raw/stage1_train').iterdir())))"
```

## Split de dados

O split treino/validação/teste (70/15/15) é estratificado por modalidade de
imagem (grayscale/fluorescência vs. colorida/histologia).

Para reproduzir do zero (opcional):
```powershell
uv run python -m scripts.split_dataset
```

## Treinar

Modelo final: **ResUNet com cabeça de 3 classes (fundo/interior/borda) +
decodificação watershed**.

```powershell
uv run python -m scripts.compute_class_weights   # gera pesos de classe (balanceamento)
uv run python -m scripts.train_dsb2018_3class    # treina o modelo final
```

Também disponível, o baseline de segmentação binária da Parte 1:
```powershell
uv run python -m scripts.train_dsb2018_baseline
```

## Avaliar

```powershell
uv run python -m scripts.evaluate_dsb2018_3class     # modelo final
uv run python -m scripts.evaluate_instance_map        # baseline
```

M�trica: mAP de instância (matching guloso por IoU decrescente, limiares
0.50 a 0.95) e erro absoluto de contagem. Ver `src/metrics/instance_matching.py`.

## Checkpoint do modelo final

Pesos treinados (ResUNet 3 classes) incluídos no repositório em `outputs/resunet_3class.pt`.

## Inferência em uma imagem nova

Abra `inferencia.ipynb`, edite a variável `IMAGE_PATH` para o caminho da
imagem desejada. O código carrega o
checkpoint acima e devolve a máscara de instâncias colorida e a contagem
de núcleos detectados.

## Estrutura do repositório

```
src/
├── datasets/        # SyntheticEllipseDataset, DSB2018Dataset, DSB2018ThreeClassDataset
├── models/          # ResUNet
├── metrics/         # IoU/Dice, mAP de instância (matching guloso)
├── postprocess/      # extração ingênua de instância, mosaico/tiling
└── postprocessing/   # decodificação watershed (Trilha A)

scripts/             # scripts de treino, avaliação e visualização
data/splits/          # split.json (versionado)
outputs/              # checkpoints (.pt, não versionados) e figuras de resultado
```

## AI_LOG

Ver `AI_LOG.md` para o registro de uso de ferramentas de IA neste projeto.