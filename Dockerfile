FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS dev
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ninja-build git gdb \
      python3 python3-pip python3-venv ca-certificates curl gpg vim \
 && rm -rf /var/lib/apt/lists/*

# Modern CMake from Kitware's official apt repo (replaces the old 22.04 one)
RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
      > /etc/apt/sources.list.d/kitware.list \
 && apt-get update && apt-get install -y --no-install-recommends cmake \
 && rm -rf /var/lib/apt/lists/*