FROM python:3.9-slim
RUN mkdir /PneumonIA
WORKDIR /PneumonIA
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /PneumonIA

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 80
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=80", "--server.address=0.0.0.0"]
