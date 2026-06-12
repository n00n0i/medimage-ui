FROM nvidia/cuda:12.8.1-base-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY monitor.py /monitor.py
CMD ["python3", "/monitor.py"]