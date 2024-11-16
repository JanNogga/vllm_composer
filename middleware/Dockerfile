FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py config.yml secrets.yml util.py ./

EXPOSE 9000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
