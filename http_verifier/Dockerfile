FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUTE_HARNESS_ARTIFACT_DIR=/profiles \
    CUTE_HARNESS_PYTHON=/usr/bin/python3 \
    CUTE_HARNESS_NSYS=/usr/local/bin/nsys \
    CUTE_DSL_CACHE_DIR=/cache/cute \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        nsight-systems-2025.3.2 \
        python3 \
        python3-pip \
    && python3 -m pip install --break-system-packages \
        "torch==2.11.0" \
        "nvidia-cutlass-dsl[cu13]==4.5.1" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY . /service
RUN python3 -m pip install --break-system-packages .

RUN useradd --create-home --uid 10001 harness \
    && mkdir -p /profiles /cache \
    && chown -R harness:harness /profiles /cache
USER harness

EXPOSE 18080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18080/healthz', timeout=3)"]
CMD ["python3", "-m", "cute_harness.app"]
