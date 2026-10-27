# Evil Pugs in a container: `docker compose up` (see compose.yaml). CUDA 12.8 to match the torch cu128 wheels.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
# gcc + libc headers: Triton (the fast linear-attention kernels) compiles a small launcher at runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements.txt .
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --torch-backend=cu128 --no-cache -r requirements.txt
COPY qwen_neurons ./qwen_neurons
COPY docker/entrypoint.sh /usr/local/bin/evil-pugs

# the base image already has an unprivileged user "ubuntu" (uid 1000)
RUN mkdir -p /data /hf-cache && chown ubuntu:ubuntu /data /hf-cache
USER ubuntu
# inside the container the one GPU it was given is cuda:0; the host picks which one (QN_GPU in compose)
ENV PATH=/opt/venv/bin:$PATH QN_HOST=0.0.0.0 QN_PORT=8765 QN_DEVICE=cuda:0 QN_DATA_DIR=/data QN_NO_BROWSER=1 \
    QN_MODEL=/model HF_HOME=/hf-cache
EXPOSE 8765
VOLUME ["/data"]
ENTRYPOINT ["evil-pugs"]
