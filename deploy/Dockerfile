# Dockerfile (multi-stage for ARM64)

# 1. Builder stage: compile Rust app with whisper-rs bindings
FROM --platform=$BUILDPLATFORM rust:1.86-slim AS builder

# Install dependencies for whisper-rs (libffi, cmake, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang cmake libffi-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Cargo files and source code
COPY Cargo.toml Cargo.lock /app/
COPY src /app/src

# Build the application in release mode
RUN cargo build --release

# 2. Runtime stage: minimal image for running
FROM --platform=linux/arm64 ubuntu:25.04 AS runtime

# (Optionally install libffi for whisper-rs if required at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi7 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy the compiled binary
COPY --from=builder /app/target/release/whisper_ws /usr/local/bin/whisper_ws

# Expose WebSocket port (example 3030)
EXPOSE 3030

# Default command: run with environment var for model path
CMD ["whisper_ws"]
