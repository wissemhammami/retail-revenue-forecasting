FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY app/ app/
COPY configs/ configs/
COPY dashboard_data/ dashboard_data/
COPY .streamlit/ .streamlit/

EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
