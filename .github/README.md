# color_prediction_app

## Структура проекта

```sh
.
├── app.py
├── data
│   ├── classes
│   ├── test
│   ├── train
├── models
├── notebooks
│   └── first.ipynb
├── requirements.txt
├── segmentation
│   └── opencv_segmentation.py
└── utils
    ├── create_dataset.py
    ├── create_jsons.py
    └── visualize_tools.py
```


В директорию `models` нужно расположить модель `scripted_model.py` и веса к ней ``
В директоию `data/classes` нужно расположить json файлы с названиями классов и их нумерацией и наоборот
В директорию `data/test` нужно расположить изображения с лего фигурками для демонстраций в лендинге
В директорию `data/train` нужно расположить изображения для обучения

## Запуск лендинга для инференса модели

```bash
python3 app.py
```

Лендинг разворачивается в http://127.0.0.1:7860