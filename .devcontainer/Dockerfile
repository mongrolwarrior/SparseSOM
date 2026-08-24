# Dev container — CUDA 12.8 devel + IDE tools.
# Source is bind-mounted at /workspace by VS Code; nothing is compiled here.
# The release image (multi-stage, slim runtime) lives in the repo-root Dockerfile.

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# GCC-12 host compiler to avoid nvcc + GCC-13 _Float32 incompatibility.
# clangd provides IntelliSense (reads compile_commands.json from cmake build).
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build git ca-certificates \
        gcc-12 g++-12 \
        clangd \
        gdb \
        python3 python3-pip python3-venv \
        wget curl \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-12 CXX=g++-12
WORKDIR /workspace
