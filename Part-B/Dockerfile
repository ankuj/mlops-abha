# Build context MUST be MLOps-Abha (not Part-B). From MLOps-Abha/ run:
#   docker build -f Part-B/Dockerfile -t cupguard .
# Run:
#   docker run --rm -p 8501:8501 cupguard
#   Streamlit → http://localhost:8501

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY Part-A/pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY Part-B/cupguard/ ./cupguard/

WORKDIR /app/cupguard/src

EXPOSE 8501

CMD ["streamlit", "run", "worldcup_agent_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
