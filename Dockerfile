# Build stage
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    git \
    curl \
    python3 \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Bazelisk
RUN npm install -g @bazel/bazelisk

# Set working directory
WORKDIR /workspace

# Copy source files
COPY . .

# Build JSIR
RUN bazelisk build //maldoca/js/ir:jsir_gen

# Runtime stage
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    libstdc++6 \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /workspace/bazel-bin/maldoca/js/ir/jsir_gen /usr/local/bin/jsir_gen

# Copy examples
COPY --from=builder /workspace/maldoca/js/ir/conversion/tests /examples

WORKDIR /workspace

CMD ["/bin/bash"]
