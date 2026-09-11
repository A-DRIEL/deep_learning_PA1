# Deep Learning PA1 — Segmentação de Instâncias

## Setup do ambiente

Este projeto usa [uv](https://docs.astral.sh/uv/) para gerenciar dependências.

1. Instale o uv (se ainda não tiver):
```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Clone o repositório e entre na pasta:
```powershell
   git clone <url-do-repo>
   cd deep_learning_PA1
```

3. Sincronize as dependências, escolhendo conforme seu hardware:

   **Com GPU NVIDIA (CUDA 12.8+):**
```powershell
   uv sync --extra gpu
```

   **Sem GPU (CPU-only):**
```powershell
   uv sync --extra cpu
```

4. Confirme que funcionou:
```powershell
   uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
   em CPU, `cuda.is_available()` deve mostrar `False`

## Download dos dados

Usamos o dataset DSB2018 (BBBC038v1), via Kaggle API.

1. Autentique no Kaggle (abre o navegador):
```powershell
   uv run kaggle auth login
```
   Se ainda não aceitou as regras da competição, acesse
   https://www.kaggle.com/competitions/data-science-bowl-2018/rules
   e clique em "I Understand and Accept" antes de continuar.

2. Baixe e extraia os dados:
```powershell
   New-Item -ItemType Directory -Force -Path data\raw
   uv run kaggle competitions download -c data-science-bowl-2018 -f stage1_train.zip -p data\raw
   uv run kaggle competitions download -c data-science-bowl-2018 -f stage1_test.zip -p data\raw
   Expand-Archive -Path "data\raw\stage1_train.zip" -DestinationPath "data\raw\stage1_train" -Force
   Expand-Archive -Path "data\raw\stage1_test.zip" -DestinationPath "data\raw\stage1_test" -Force
   Remove-Item data\raw\stage1_train.zip, data\raw\stage1_test.zip
```

3. Confirme que baixou certo (deve mostrar 670):
```powershell
   uv run python -c "from pathlib import Path; print(len(list(Path('data/raw/stage1_train').iterdir())))"
```

## Split de dados

O split treino/validação/teste é estratificado por modalidade de imagem
(grayscale/fluorescência vs. colorida), inferida via variância de cor
entre canais RGB (ver `scripts/inspect_modalities.py` para a análise
que motivou o threshold escolhido).

O split já está versionado em `data/splits/split.json` para manter os resultados
comparáveis entre diferentes execuções.

Para validar que o split está íntegro:
```powershell
uv run python -m scripts.validate_split
```

## Testes de sanidade (Parte 0)

Confirma que o pipeline completo funciona, com dataset sintético:
```powershell
uv run python -m scripts.sanity_check_synthetic
```
Treina em poucos minutos, IoU/Dice próximos de 1.0.

## Treino no dado real (Parte 1)

```powershell
uv run python -m scripts.train_dsb2018_baseline
```

**Atenção**: em CPU, isso é significativamente mais lento que em GPU.
Recomendado rodar em GPU local, ou usar Google Colab / Kaggle Notebooks
se não tiver GPU disponível.

## Visualizações

```powershell
uv run python -m scripts.visualize_synthetic   # amostras do dataset sintético
uv run python -m scripts.visualize_dsb2018     # amostras do dataset real
```