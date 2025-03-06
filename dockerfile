# Use Python 3.8-slim as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and data files
COPY . .

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the server
CMD ["python", "main.py"]