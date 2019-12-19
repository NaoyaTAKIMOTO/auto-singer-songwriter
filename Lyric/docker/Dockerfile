# Use an official Python runtime as a base image
FROM python:3.6
ENV USER=app
ENV HOME=/home/$USER
ENV PROJECT=$HOME/lyric
#Add user for app
RUN \
useradd --user-group --create-home --shell /bin/false $USER;\
mkdir $PROJECT;
WORKDIR $PROJECT
COPY src/  $PROJECT/
COPY requirements.txt $PROJECT/
RUN apt-get update \
    && apt-get install -y mecab \
    && apt-get install -y libmecab-dev \
    && apt-get install -y mecab-ipadic-utf8
RUN pip install -r requirements.txt --no-cache-dir;

EXPOSE 5000
CMD python run_server.py
