FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl vim && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
RUN python -m pip install --upgrade pip setuptools wheel
CMD ["bash"]
