# Use a specific, stable version of Python for reliability
FROM python:3.10.13-slim

# Set environment variables to ensure logs are not buffered and are shown immediately
ENV PYTHONUNBUFFERED True
ENV FLASK_APP=app.py

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install the Python libraries from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy all your application files, including the large models
COPY . .


CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app