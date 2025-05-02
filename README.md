# whisper.rs


A high-performance Rust implementation of OpenAI's Whisper live streaming speech recognition model, designed for real-time transcription over WebSocket. This project leverages [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for efficient inference and provides a robust, configurable server for speech-to-text applications.

---

## Features

- **Real-time streaming transcription** via WebSocket API
- **Voice Activity Detection (VAD)** using energy-based algorithm for efficient segmentation
- **Configurable model, threading, and API key** via environment variables
- **Graceful shutdown** and robust error handling
- **Health check endpoint** for monitoring
- **Docker support** for easy deployment

---

## Project Structure

```
.
├── bin/
│   ├── basic_vad.rs, main_vad.rs, main_webrtc.rs, main.rs  # Main binaries
├── src/
│   ├── lib.rs
│   ├── config/
│   │   ├── service_config.rs, service_params.rs
│   ├── service/
│   │   ├── whisper_service.rs
│   │   └── shared_state/
│   │       └── shared_state.rs
│   ├── vad/
│   │   └── energy_vad.rs
│   └── whisper/
│       ├── whisper_callback.rs, whisper_config.rs, whisper_helper.rs
├── models/                    # Pretrained Whisper models and audio samples
├── deploy/
│   └── Dockerfile             # Docker deployment
├── Cargo.toml                 # Rust crate manifest
├── LICENSE
├── README.md                  # This file
└── ...
```

---

## Getting Started

### Prerequisites

- Rust (stable, recommended latest)
- [whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp/tree/main) - compatible model files (see `models/`)
- (Optional) Docker

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/zs-dima/whisper.rs.git
   cd whisper.rs
   ```

2. **Download a Whisper model:**
   Place a `.bin` model file (e.g., `ggml-base.en-q5_1.bin`) in the `models/` directory. You can use the provided models or download from [whisper.cpp huggingface](https://huggingface.co/ggerganov/whisper.cpp/tree/main).

3. **Build the project:**
   ```sh
   cargo build --release
   ```

---

## Configuration

Configuration is managed via environment variables. Defaults are provided for all options.

| Variable             | Description                        | Default                        |
|----------------------|------------------------------------|--------------------------------|
| `PORT`               | Server port                        | `3030`                         |
| `API_KEY`            | API key for WebSocket auth         |                                |
| `WHISPER_MODEL_NAME` | Path to Whisper model file nme     | `ggml-base.en-q5_1.bin`        |
| `WHISPER_MODEL_URL`  | URL to Whisper a model             |  [whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$WHISPER_MODEL_NAME?download=true)          |
| `WHISPER_THREADS`    | Number of inference threads        | `min(4, num_cpus)`             |
| `SAMPLE_THRESHOLD`   | Min samples before flush (silence) | `64000` (4s at 16kHz)          |

Example:
```sh
$env:PORT=8080
$env:API_KEY="your-secret-key"
$env:WHISPER_MODEL_NAME="ggml-base.en-q5_1.bin"
$env:WHISPER_THREADS=4
$env:SAMPLE_THRESHOLD=32000
```

---

## Running the Server

```sh
cargo run --release --bin main
```

The server will start and listen on the configured port (default: 3030).

---

## API

### Health Check

- **GET** `/health`
- **Response:** `"OK"`

### WebSocket Transcription

- **Endpoint:** `/ws?api-key=YOUR_API_KEY`
- **Protocol:** WebSocket
- **Audio Format:** Little-endian `f32` PCM, 16kHz, mono, streamed as binary frames

#### Example WebSocket Session

1. **Connect:**
   ```
   ws://localhost:3030/ws?api-key=YOUR_API_KEY
   ```

2. **Send:** Binary audio data in chunks (each chunk is a multiple of 4 bytes, representing `f32` samples).

3. **Receive:** JSON messages:
   ```json
   {"text": "recognized transcript", "seq": 1}
   ```

---

## Voice Activity Detection (VAD)

The server uses an energy-based VAD ([`vad::energy_vad`](src/vad/energy_vad.rs)) to segment speech and trigger transcription flushes. Energy VAD works in pair with Whisper sentences recognition. Silence is detected when the energy in the last 200ms falls below 35% of the average energy.

---

## Docker

Build and run with Docker:

```sh
# Build
docker build -t whisper-rs .
# Run
# (Mount your models directory for access to model files)
docker run -p 3030:3030 -v ${PWD}/models:/app/models whisper-rs
```

---

## Models

Place Whisper model files in the `models/` directory. Supported formats: `.bin` (ggml), `.onnx` (for ONNX-based binaries).

---

## Development

- Code is organized as a library with main binaries in `bin/`.
- Main server logic is in `service::whisper_service`.
- VAD logic is in `vad::energy_vad`.
- Configuration is handled by `config::service_config`.

---

## Roadmap

- ✅ Implement lightweight, effective Energy VAD that works in pair with Whisper sentences recognition
- [ ] Detailed configuration options for VAD, threading, etc
- [ ] Add Silero VAD as an option
- [ ] Provide WebRTC in addition to WebSocket
- [ ] Add authentication/authorization options
- [ ] Improve error reporting and logging
- [ ] Add live demo client (web UI)
- [ ] Expand Docker deployment options (multi-arch, GPU)
- [ ] Add CI/CD pipeline and automated tests
- [ ] Publish to Docker Hub and etc

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [whisper-rs](https://github.com/tazz4843/whisper-rs)
- [Axum](https://github.com/tokio-rs/axum)