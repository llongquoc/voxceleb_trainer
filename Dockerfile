FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt
COPY ./ ./
EXPOSE 8080

CMD ["python" , "server.py"]