# Build stage
FROM rust:latest AS builder
WORKDIR /workspace

# Cache dependencies:
COPY Cargo.toml Cargo.lock
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src
# Copy the full source code and rebuild
COPY . .
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENV RUST_LOG=info
COPY --from=builder /workspace/target/release/vllm_middleware /usr/local/bin/vllm_middleware
EXPOSE 9000
ENTRYPOINT ["/usr/local/bin/vllm_middleware", "9000"]
