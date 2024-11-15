FROM python:3.10-slim

WORKDIR /app
COPY app.py requirements.txt config.yml secrets.yml util.py ./

RUN pip install --no-cache-dir -r requirements.txt uvicorn

EXPOSE 9000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
