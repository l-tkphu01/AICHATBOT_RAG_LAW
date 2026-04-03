# ==========================================
# STAGE 1: Build Dependencies (Builder)
# ==========================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /install

# Cài đặt build tools (gcc, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Build wheels file thay vì install thẳng để giảm dung lượng
COPY requirements.backend.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.backend.txt

# ==========================================
# STAGE 2: Minimal Runtime (Runner)
# ==========================================
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# (Tùy chọn) cài đặt thư viện chạy native (VD: libpq cho psycopg2) & curl cho healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packaged wheels từ Stage 1
COPY --from=builder /wheels /wheels
COPY requirements.backend.txt .
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Phân quyền cho Non-Root User (Best Practice Security)
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Copy source code
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

# Chown thư mục app
RUN chown -R raguser:raguser /app
USER raguser

EXPOSE 8000

# Container entry point
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
