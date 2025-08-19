FROM python:3.10.13-slim
ENV PYTHONUNBUFFERED True
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Use the simple, robust Gunicorn command pointing to app.py
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]