# Классификатор интентов чат-бота
Этот проект реализует классификацию интентов в чат-боте с использованием модели k-ближайших соседей (kNN) и текстовых эмбеддингов. Такой подход позволяет быстро обновлять и модифицировать интенты без необходимости повторного обучения модели, что особенно важно для динамично меняющихся бизнес-требований.

Проект разработан совместно с компанией Сравни и решает прикладную бизнес задачу.

## Команда
- Азаматов Ильдар - Сбор данных, реализация бейзлайн решения, коммуникации с компанией Сравни
- Михаил Репкин - Эксперименты с текстовыми эмбеддингами (bge-m3, e5), обучение с contrastive loss
- Анастасия Изюмова - Эксперименты с текстовыми эмбеддингами (LaBSE), обучение с triplet loss

## Содержание репозитория
- [src](src) - Исходные коды сервиса на стримлите, вспомогательные скрипты
- [notebooks](notebooks) - Ноутбуки с экспериментами
- [data](data) - Данные (исходные и обработанные). В данной папке содержится датасет с интентами, однако валидационный датасет с фразами пользователя не залит в гит по соображениям безопасности. Для получения доступа напишите в tg @eelduck



## Эксперименты
Для данного демо были проведены несколько экспериментов с моделями multilingual-e5-large, bge-m3 и LaBSE. Ноутбуки с подсчетом метрик и finetuning моделей:

BGE-M3, E5: [ноутбук с EDA, Finetuning, Inference](https://github.com/DmitryChatBotov/intent-classifier/blob/main/notebooks/m-e5_bge-m3_experiments.ipynb)

LaBSE: [finetuning](https://github.com/DmitryChatBotov/intent-classifier/blob/main/notebooks/LaBSE_finetune.ipynb), [inference](https://github.com/DmitryChatBotov/intent-classifier/blob/main/notebooks/LaBSE_inference.ipynb)

### Таблица с результатами экспериментов:

| Model | F1-macro | Accuracy |
| ----- | -------- | -------- |
| Regexp-Based classification (baseline) | - |0.45|  
| m-e5-large, pretrained, knn-3 |  0.432 |  0.548 |  
|m-e5-large, fine-tuned, Online Contrastive Loss, knn-9 | <ins>0.523</ins> | <ins>0.627</ins> |  
|m-e5-large, fine-tuned, Multiple Negatives Ranking Loss, knn-7 |   0.451 |  0.617|
|bge-m3, pretrained, knn-7 | 0.472 | 0.593|
|bge-m3, fine-tuned, Online Contrastive Loss, knn-5 | 0.517|0.597|
|bge-m3, fine-tuned, Multiple Negatives Ranking Loss, knn-3| 0.437 | 0.556| 
|bge-m3, fine-tuned, CrossEntropyLoss, knn-8| **0.547** | **0.683**|
|LaBSE, pretrained, knn-18 |   0.27 | 0.4|
|LaBSE, fine-tuned, ContrastiveLoss, knn-12 |   0.4  | 0.54|
|LaBSE, fine-tuned, TripletLoss, knn-11 |  0.24  | 0.362|
## Подготовка интентов
Перед тем, как разворачивать приложение, необходимо векторизовать данные и занести их в chromadb, для этого необходимо:
- В файле __src/add_chroma_documents.py__ указать необхоидмую модель эмбеддингов, а также тэг для коллекции chromadb
- Запустить скрипт __src/add_chroma_documents.py__
- В корне проекта должна появиться папка __chromadb_data__

## Запуск демо на streamlit
![image](https://github.com/DmitryChatBotov/intent-classifier/assets/41739221/32c45d02-ee22-4e86-aaea-69e89e4bbdee)

- Установите python 3.10.11
- Создайте виртуальное окружение с помощью команды 
```bash
python -m venv .venv
```
- Активируйте окружение
    - Для Windows:
    ```bash
    .\.venv\Scripts\activate
    ``` 
    - Для Linux:
    ```bash
    source .venv/bin/activate
    ``` 

- Установите необходимые библиотеки
```bash
pip install -r requirements.txt
```

- Запустите приложение
```bash
streamlit run src/streamlit_app.py
```
