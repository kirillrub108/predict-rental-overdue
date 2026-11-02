FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY predict.py model_hgb.bin ./
EXPOSE 8000
CMD uvicorn predict:app --host 0.0.0.0 --port ${PORT:-8000}