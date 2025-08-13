# Start with a lightweight version of Python
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy ONLY the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of your application, including the model files
COPY . .

# Tell Render that your application will be listening on port 8080
EXPOSE 8080

# The command that Render will use to start your application
# It uses Gunicorn, a production-ready web server.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]