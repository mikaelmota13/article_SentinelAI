FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libxml2-dev \
    libgl1-mesa-glx \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    pyarrow \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
COPY script.py .
COPY df.parquet .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "script.py"]
