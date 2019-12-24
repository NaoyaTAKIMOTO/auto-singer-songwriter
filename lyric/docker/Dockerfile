# Use an official Python runtime as a base image
FROM python:3.6-slim-buster
ENV USER=app
ENV HOME=/home/$USER
ENV PROJECT=$HOME/lyric
#Add user for app
RUN \
useradd --user-group --create-home --shell /bin/false $USER;\
mkdir -p $PROJECT;
RUN mkdir -p $PROJECT/model/tokenizer
WORKDIR $PROJECT
COPY src/  $PROJECT/src
COPY requirements.txt $PROJECT/
RUN apt-get update \
    && apt-get install -y mecab build-essential python3-setuptools\
    && apt-get install -y libmecab-dev \
    && apt-get install -y mecab-ipadic-utf8\
    && apt-get install -y swig python3-pip

RUN mecab
RUN pip3 install mecab-python3
RUN pip3 install --upgrade pip&&pip3 install -r requirements.txt --no-cache-dir;

EXPOSE 5000
RUN python src/download_model.py
CMD python src/run_server.py
