 FROM python:3.10.13-slim
    ENV PYTHONUNBUFFERED True
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    # The final, correct command for the factory pattern
    CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "wsgi:app"]
    ```