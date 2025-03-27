# Use Python 3.8-slim as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y espeak

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U langchain-community
RUN python -m spacy download en_core_web_sm

# Copy the application code and data files
COPY . .

# Expose port 8000 for the FastAPI server
EXPOSE 80

# Command to run the server
CMD ["python", "main.py"]