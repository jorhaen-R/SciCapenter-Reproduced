FROM python:3.9-slim

# Install Java runtime for pdffigures2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre-headless \
    libfontconfig1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install Python dependencies first (for better cache use)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
