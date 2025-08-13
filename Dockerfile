# Start with a specific, stable version of Python
FROM python:3.10.13-slim

# Set environment variables to ensure logs are output correctly
ENV PYTHONUNBUFFERED True

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of your application, including the model files
COPY . .

# EXPOSE is good practice but not strictly required by Render.
# The Gunicorn command below is the critical part.
EXPOSE 10000

# The final, robust command to start your application.
# It tells Gunicorn to use the $PORT variable provided by Render.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]