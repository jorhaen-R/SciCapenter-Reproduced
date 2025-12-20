# 使用官方 Python 3.9 Slim 镜像
FROM python:3.9-slim

# Switch to Aliyun sources for speed
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# Install Java + Graphics Libraries (Critical for pdffigures2)
# unset proxies to force direct connection to domestic mirrors
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    apt-get update --fix-missing -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
    default-jre-headless \
    libfontconfig1 libxtst6 libgl1 libglib2.0-0 \
    libxrender1 libxext6 libxi6 ghostscript fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.docker.local.txt /app/requirements.txt

# Install Python deps via Tsinghua source
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    pip install --no-cache-dir -r requirements.txt \
    -i http://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --default-timeout=100

COPY . /app
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
