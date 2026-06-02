FROM python:3.12-slim
RUN ln -sf /usr/share/zoneinfo/America/Chicago /etc/localtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nano \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
# Expose Jupyter's default port
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
