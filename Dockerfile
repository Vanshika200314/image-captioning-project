# Use a specific, stable version of Python for reliability
FROM python:3.10.13-slim

# Set an environment variable to ensure logs are shown immediately
ENV PYTHONUNBUFFERED True

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Install all Python libraries from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy all your application files, including the large models
COPY . .

# The final, correct command to start the Gunicorn server.
# This "shell" form correctly uses the $PORT variable provided by Render.
# This line is the definitive fix for the port error.
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app