# Emotion Recognition from Speech Segment

Проект предлагает решение задачи классификации эмоций в отрывке речи.

## Описание

Используется нейросетевая архитектура CNN + Attention над спектрограммами аудио.
Данные обрабатываются в виде мел-спектрограмм. В проекте поддерживается как
обучение модели, так и инференс на новых аудиофайлах.

Для обучения готовой модели был выбран датасет
[CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad), в котором содержится
7442 записи голоса от 91 актера. В датасете содержится 6 различных классов -
эмоций гнева, отвращения, страха, счастья, нейтральности и грусти.

### Особенности:

- Работа с зашумлёнными аудио, использование аугментаций при обучении модели.
- Использование CNN + Transformer Encoder + FC.
- PyTorch Lightning для модульной и читаемой структуры кода.
- Поддержка конфигураций через Hydra.

### Используемые технологии

- Lightning
- Poetry
- Wandb logging
- Hydra config
- Pytest

## Работа с проектом

### Setup

Для начала работы:

1. Установите [Poetry](https://python-poetry.org/docs/):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Клонируйте репозиторий и перейдите в папку проекта:
   ```bash
   git clone https://github.com/your-username/your-project.git
   cd your-project
   ```
3. Создайте окружение и установите зависимости, активируйте окружение:
   ```bash
   poetry install
   poetry shell
   ```
4. Установите и запустите pre-commit хуки:
   ```bash
   pre-commit install
   pre-commit run -a
   ```

### Train

Для запуска обучения модели:

```bash
python src/train.py
```

### Inference

Для запуска модели в режиме предсказаний:

```bash
python src/infer.py
```

## 📂 Структура проекта:

```bash
📁 speech-emotion-recognition/      # (root)
├── 📁 .git/                        # конфигурации git (actions и др.)
├── 📁 .dvc/                        # хэши и конфиги dvc
├── 📁 conf/                        # конфиги hydra
├── 📁 data/                        # папка для хранения данных
│   └── 📁                          #
├── 📁 models/                      # сохранённые веса обученных моделей
├── 📁 scripts/                     # скрипты для деления датасета на выборки и обновления весов
├── 📁 speech_emotion_recognition/  # папка с кодом проекта
│   ├── 📄 classifier.py            # реализация классификатора с помощью torch
│   ├── 📄 data.py                  # описание датамодулей и датасетов lightning
│   ├── 📄 inference.py             # запуск предсказания модели
│   ├── 📄 loss.py                  # реализация focal loss для обучения
│   ├── 📄 model.py                 # реализация lightning module модели
│   ├── 📄 preprocessing.py         # реализация препроцессинга и аугментаций данных
│   └── 📄 train.py                 # обучение модели
├── 📁 tests/                       # тесты pytest
├── 📄 .dvcignore                   # исключения для dvc
├── 📄 .gitignore                   # исключения для Git
├── 📄 .pre-commit-config.yaml      # настройка линтеров и хуков
├── 📄 poetry.lock                  #
├── 📄 pyproject.toml               # зависимости (Poetry)
└── 📄 README.md                    # описание проекта
```
