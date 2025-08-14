# Use a specific, stable version of Python
FROM python:3.10.13-slim

# Set environment variables for stability
ENV PYTHONUNBUFFERED True
ENV FLASK_APP=app.py

# Set the working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files, including models
COPY . .

# Run the Gunicorn server, binding to the port Render provides
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app