# Демо классификатора сообщений в чат боте

## Подготовка интентов
Перед тем, как разворачивать приложение, необходимо векторизовать данные и занести их в chromadb, для этого необходимо:
- В файле __src/add_chroma_documents.py__ указать необхоидмую модель эмбеддингов, а также тэг для коллекции chromadb
- Запустить скрипт __src/add_chroma_documents.py__
- В корне проекта должна появиться папка __chromadb_data__

## Запуск приложения
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