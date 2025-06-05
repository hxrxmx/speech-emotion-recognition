# Emotion Recognition from Speech Using CNN + Transformer Encoder.

## Краткое описание

Проект предлагает решение задачи классификации эмоций в отрывке речи.
Используется нейросетевая архитектура CNN + Transformer Encoder над
спектрограммами аудио. Поддерживается как обучение модели, так и инференс на
новых аудиофайлах.

<details>
<summary>Цели</summary>

- Реализация архитектуры на основе спектрограмм с использованием сверточных и
  attention-слоёв для решения поставленной задачи.
- Реализация удобного пайплайна для обучения и инференса с использованием
  современных инструментов.

</details>

<details>
<summary>Особенности</summary>

- Предобученные версии модели используются для классификации англоязычных
  отрывков речи.
- Работа с зашумлёнными аудио, использование аугментаций при обучении модели.
- Использование CLI интерфейса для загрузки датасета и начала обучения модели.
- Использование CNN + Transformer Encoder.
- Использование Lightning для реализации модульной читаемой структуры кода.
- Логгирование с помощью wandb.

</details>

<details>
<summary>Используемые технологии</summary>

- Lightning
- Poetry
- Wandb logging
- Hydra config
- Pytest

</details>

## Архитектура

При обучении в начале пайплайна аудиосегменты обрезаются до максимальной длины в
5 секунд и подвергаются аугментациям: сдвигам по времени до половины длины
аудиосегмента, применению low-pass (до 4 кГц) и high-pass (до 800 Гц) фильтров,
добавлению шума разных типов, питчу частоты от -2 до +3 полутонов и изменению
громкости от -6 до +12 дБ. Затем преобразуются в Mel-спектрограммы, подвергаются
триму шума менее 7 дБ от минимального уровня. Итогом препроцессинга аудиозаписей
являются спектрограммы с частотным разрешением в 256 пикселей и временным
в 1024. Для извлечения эмбеддингов из спектрограмм используется CNN. В ней число
каналов преобразуется, как 1 $\xrightarrow{\text{ker}=7\times7}$ 32
$\xrightarrow{\text{ker}=3\times3}$ 64 $\xrightarrow{\text{ker}=3\times3}$ 128
$\xrightarrow{\text{ker}=3\times3}$ 256 $\xrightarrow{\text{ker}=3\times3}$ 512.
Уменьшение временного и частотного разрешения реализовано с помощью MaxPool
слоев с ядрами (4, 1) и (2, 2). На выходе из CNN получается последовательность
векторов эмбеддингов размера 512 длиной 256. Далее следует энкодер трансформера
с параметрами $dim_\text{embed} = 512$, $n_\text{head} = 4$,
$dim_\text{feedforward} = 1024$, $n_\text{layers} = 6$. После следует AvgPool по
длине последовательности и линейный слой $512 \times n_\text{classes}$.

## Данные для обучения

Для обучения готовой модели был выбран датасет
[CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad), в котором содержится
7442 записи англоязычных реплик от 91 актера. В датасете содержится 6 различных
классов - эмоций гнева (label 0), отвращения (label 1), страха (label 2),
счастья (label 3), нейтральности (label 4) и грусти (label 5) [нумерация в
алфавитном порядке в переводе на английский].

## Оценка качества моделей

Для оценки качества используюся метрики accuracy и f1-score, которые во время
обучения логируются с помощью wandb. Удалось обучить модель до $\sim 0.65$
accuracy и f1-score на тестовой выборке.

# Работа с проектом

## Setup

Для начала работы:

1. Установите [Poetry](https://python-poetry.org/docs/):

   ```bash
   pip install poetry
   ```

2. Клонируйте репозиторий и перейдите в папку проекта:

   ```bash
   git clone https://github.com/hxrxmx/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

3. Создайте окружение и установите зависимости, активируйте окружение:

   ```bash
   poetry install
   source $(poetry env info --path)/bin/activate
   ```

   (На windows для активации окружения нужно использовать `poetry env activate`.
   Это вернет команду, с помощью которой можно активировать среду)

## Train

### Загрузка и подготовка данных

Используемый в проекте датасет располагается на kaggle. Можно скачать его
вручную непосредственно с kaggle и поместить папку CREMA-D в `(root)/data/` или
воспользоваться скриптом скачивания. В последнем случае используется kaggle api,
для использования которого необходима авторизация. Скрипт скачивания датасета
нужно запустить из директории `(root)/scripts/download/`:

```bash
python download_data.py
```

<details> <summary>Авторизация на kaggle</summary>
  <blockquote>

Если ранее kaggle api не использовался, зарегестрируйтесь на kaggle, перейдите в
настройки, и выберите "Create New Token". Это запустит скачивание `kaggle.json`
файла, который нужно поместить по пути `~/.kaggle/` (или другому пути, который
указан во всплывающем окне [например, `~/.config/kaggle/`]).

  </blockquote>
</details>
<br>

После скачивания нужно разбить датасет на выборки и классы. Кроме того, для
корректного подсчета взвешенного loss при обучении необходимо рассчитать веса
для классов. Для этого из директории `(root)/scripts/preparation/` нужно
запустить скрипт разбиения датасета, а затем записать в конфиг веса для классов:

```bash
python split_dataset.py
python update_cls_weights.py
```

### Обучение модели

Для запуска обучения модели необходимо вызвать из директории
`(root)/speech_emotion_recognition/`:

```bash
python train.py
```

## Inference

### Загрузка предобученной модели

Для скачивания предобученной модели перейдите в папку `(root)/scripts/download/`
и запустите:

```bash
python download_model.py
```

### Запуск в режиме предсказаний

Перейдите в `(root)/speech_emotion_recognition/` и запустите:

```bash
python inference.py --ckpt_path="relative/path/to/model.ckpt" --paths="in quotes relative paths separated by spaces"
```

<details>
  <summary><i>Дополнительно</i></summary>
  <blockquote>

_По умолчанию (при запуске
`python inference.py --paths="path1.wav path2.wav ..."`) используется модель,
указанная в hydra-конфиге (`config.inference.ckpt_path`)._

_Допускается запуск с другим конфигом, для этого нужно указать при запуске
`--config_path="relative/path/to/config/dir"` и `--config_name="config_name"`_

  </blockquote>
</details>

<details>
  <summary><i>Пример запуска</i></summary>

```bash
python inference.py --paths="../data/CREMA-D-split/test/HAP/1004_TIE_HAP_XX.wav ../data/CREMA-D-split/test/DIS/1005_IWL_DIS_XX.wav ../data/CREMA-D-split/test/SAD/1006_DFA_SAD_XX.wav" --ckpt_path="../models/model-epoch=78-val_loss=0.7900-val_acc=0.674.ckpt"
```

</details>

## Test

Для проверки метрик модели на тестовой выборке нужно запустить из
`(root)/speech_emotion_recognition/`:

```bash
python test.py --ckpt_path="relative/path/to/model.ckpt"
```

<details>
  <summary><i>Дополнительно</i></summary>
  <blockquote>

_По умолчанию (при запуске `python test.py"`) используется модель, указанная в
hydra-конфиге (`config.inference.ckpt_path`)._

_Допускается запуск с другим конфигом, для этого нужно указать при запуске
`--config_path="relative/path/to/config/dir"` и `--config_name="config_name"`_

  </blockquote>
</details>

<details>
  <summary><i>Пример запуска</i></summary>

```bash
python test.py --ckpt_path="../models/model-epoch=78-val_loss=0.7900-val_acc=0.674.ckpt"
```

</details>

## Model production packaging

Во время обучения три модели, лучше показавшие себя на валидации сохраняются в
`(root)/models/`. Для сохранения модели в других форматах -- onnx и TensorRT,
нужно запустить из `(root)/scripts/production_packaging/`:

```bash
python convert.py --ckpt_path="relative/path/to/saved/model.ckpt"
```

<details>
  <summary><i>Дополнительно</i></summary>
  <blockquote>

_По умолчанию веса в форматах .onnx и .pth сохраняются в той же директории и с
тем же исенем, что и выбранная модель .ckpt, но допускается сохранение их в
другом месте. Для этого нужно указать
`--onnx_path="relative/path/for/model.onnx"` и
`--trt_path="relative/path/for/model.pth"`_

_Допускается запуск с другим конфигом, для этого нужно указать при запуске
`--config_path="relative/path/to/config/dir"` и `--config_name="config_name"`_

  </blockquote>
</details>

<details>
  <summary><i>Пример запуска</i></summary>

```bash
python convert.py --ckpt_path="../../models/model-epoch=78-val_loss=0.7900-val_acc=0.674.ckpt"
```

</details>

## Проверка качества и тесты кода

Для запуска pre-commit хуков:

```bash
pre-commit install
pre-commit run -a
```

Для запуска pytest нужно запустить из `(root)/`:

```bash
pytest tests/
```

Можно проверить покрытие кода тестами, установив флаг
`--cov=speech_emotion_recognition`.

## 📂 Структура проекта

```bash
📁 speech-emotion-recognition/        # (root)
├── 📁 .github/workflows/
├── 📁 .dvc/
├── 📁 conf/                          # конфиги hydra
├── 📁 data/                          # директория для хранения данных
│   └── 📁
├── 📁 logs/wandb/                    # логи wandb
├── 📁 models/                        # сохранённые веса обученных моделей
│
├── 📁 scripts/
│   ├── 📁 download
│   │   ├── 📄 download_data.py       # скачивание датасета
│   │   └── 📄 download_model.py      # скачивание предобученной модели
│   ├── 📁 preparation
│   │   ├── 📄 split_dataset.py       # разделение датасета
│   │   └── 📄 update_cls_weights.py  # обновление весов классов
│   └── 📁 production_packaging
│       └── 📄 convert.py             # сохранение весов модели в .onnx и .pth
│
├── 📁 speech_emotion_recognition/
│   ├── 📁 core
│   │   ├── 📄 classifier.py          # torch классификатор спектрограмм
│   │   ├── 📄 loss.py                # focal loss
│   │   └── 📄 model.py               # lightning модель
│   │
│   ├── 📁 data
│   │   ├── 📄 augmentations.py       # аугментации
│   │   ├── 📄 data.py                # lightning датамодули и датасеты
│   │   └── 📄 preprocessing.py       # препроцессинг
│   │
│   ├── 📁 inference
│   │   ├── 📄 loading.py             # загрузка модели
│   │   └── 📄 preprocessing.py       # реализация препроцессинга для предсказаний
│   │
│   ├── 📁 utils
│   │   └── 📄 plotting.py            # колбэк с построением графика локально
│   │
│   ├── 📄 predict.py                 # запуск модели в режиме предсказаний
│   ├── 📄 test.py                    # измерение метрик на тестовых данных
│   └── 📄 train.py                   # обучение модели
│
├── 📁 tests/                         # тесты pytest
├── 📄 .pre-commit-config.yaml
├── 📄 poetry.lock
├── 📄 pyproject.toml
├── 📄 pytest.ini
└── 📄 README.md
```
