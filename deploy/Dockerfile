# Dockerfile (multi-stage for ARM64)

# 1. Builder stage: compile Rust app with whisper-rs bindings
FROM --platform=$BUILDPLATFORM rust:1.86-slim AS builder

# Install ALL build dependencies, including make
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang cmake libffi-dev pkg-config make && \
    rm -rf /var/lib/apt/lists/*

# Install rustfmt for the toolchain
# RUN rustup component add rustfmt

WORKDIR /app

# Copy Cargo files and source code
COPY Cargo.toml Cargo.lock /app/
COPY src /app/src
COPY bin /app/bin

# Build the application in release mode
RUN cargo build --release

# 2. Runtime stage: minimal image for running
FROM --platform=linux/arm64 ubuntu:25.04 AS runtime

# (Optionally install libffi for whisper-rs if required at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Add entrypoint script
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the compiled binary
COPY --from=builder /app/target/release/whisper-server /usr/local/bin/whisper-server

# Expose WebSocket port (example 3030)
EXPOSE 3030

# Entrypoint handles model download and runs app
ENTRYPOINT ["/entrypoint.sh"]

# # Default command: run with environment var for model path
# CMD ["whisper-server"]
