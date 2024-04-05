FROM python:3.11-bookworm

WORKDIR /src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src src/
COPY project_consts.py .

EXPOSE 8501

CMD cd src && \
    streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0