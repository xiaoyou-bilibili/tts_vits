FROM registry.xiaoyou.host/nvidia/cuda:torch-1.9.0
WORKDIR /code
COPY . .
RUN apt update && apt install portaudio19-dev python3-pyaudio libsndfile1 -y && pip3 install -r requirements.txt -i https://nexus.xiaoyou.host/repository/pip-hub/simple
EXPOSE 7001
CMD ["python3","main.py"]