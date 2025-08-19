# Use a specific, stable version of Python
FROM python:3.10.13-slim

# Set environment variables for stability
ENV PYTHONUNBUFFERED True

# Set the working directory
WORKDIR /app

# Copy requirements file first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files, including models
COPY . .

# The final, correct command to start the Gunicorn server
# This "shell" form correctly uses the $PORT variable provided by Render
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app