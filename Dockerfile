FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
    && apt-get clean

WORKDIR /app

COPY requirements.txt ./
COPY script.py ./
COPY X_resampled_ho.parquet ./
COPY Y_resampled_ho.parquet ./

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD ["python", "script.py"]
