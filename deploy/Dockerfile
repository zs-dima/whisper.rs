# Dockerfile (multi-stage for ARM64)
# 
# docker build --no-cache --pull -f deploy/Dockerfile -t whisper-rs .
# docker run --rm -e SERVICE_PORT=3030 -p 3030:3030 -e WHISPER_MODEL_NAME=ggml-base.en-q5_1.bin whisper-rs
# docker run -it --entrypoint /bin/bash whisper-rs
# which whisper-server
# ls -l /usr/local/bin/whisper-server
# ls -l /local/bin/whisper-server

# 1. Builder stage: compile Rust app with whisper-rs bindings
FROM --platform=$BUILDPLATFORM rust:1.86-slim AS builder

# Install ALL build dependencies, including make
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang cmake libffi-dev make pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Cargo files and source code
COPY Cargo.toml Cargo.lock /app/
COPY src /app/src
COPY bin /app/bin

# Build the application in release mode
RUN cargo build --release

# 2. Runtime stage: minimal image for running
FROM --platform=$BUILDPLATFORM ubuntu:25.04 AS runtime

# (Optionally install libffi for whisper-rs if required at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libffi8 wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Add entrypoint script
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the compiled binary
COPY --from=builder /app/target/release/whisper-server /bin/whisper-server

# Expose WebSocket port (example 3030)
ENV SERVICE_PORT=3030
EXPOSE 3030

# Entrypoint handles model download and runs app
ENTRYPOINT ["/entrypoint.sh"]

# # Default command: run with environment var for model path
# CMD ["whisper-server"]
