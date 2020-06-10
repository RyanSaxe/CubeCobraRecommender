FROM python:3.7

RUN mkdir -p /recommender
WORKDIR /recommender
COPY ml_files/ ml_files/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY web/requirements.txt web/requirements.txt
RUN pip install -r web/requirements.txt --no-cache-dir

COPY web/ web/
COPY src/ src/
