# Build stage
FROM rust:latest AS builder
WORKDIR /workspace
COPY . .
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=builder /workspace/target/release/vllm_middleware /usr/local/bin/vllm_middleware
EXPOSE 9000
ENTRYPOINT ["/usr/local/bin/vllm_middleware", "9000"]
