FROM python:3.6-slim-buster
ENV USER=app
ENV HOME=/home/$USER
ENV PROJECT=$HOME/voice
#Add user for app
RUN \
useradd --user-group --create-home --shell /bin/false $USER;\
mkdir $PROJECT;
WORKDIR $PROJECT
COPY src/  $PROJECT/
RUN apt-get update && apt-get install -y vim git make gcc wget curl cmake
RUN git clone https://github.com/espnet/espnet.git
RUN apt-get install -y libsndfile1-dev
RUN apt-get install -y g++ automake autoconf unzip sox gfortran libtool subversion python2.7
RUN pip install sentencepiece torch flask
RUN pip install -q parallel_wavegan PyYaml unidecode ConfigArgparse g2p_en nltk
RUN pip install cython chainer ipython wavio editdistance
RUN apt-get install -y open-jtalk open-jtalk-mecab-naist-jdic
WORKDIR $PROJECT
RUN mkdir tools && cd tools && git clone https://github.com/r9y9/hts_engine_API.git
ENV TOOLS=$PROJECT/tools
RUN cd $TOOLS/hts_engine_API/src && ./waf configure && ./waf build install
RUN cd $TOOLS && git clone https://github.com/r9y9/open_jtalk.git
RUN mkdir -p $TOOLS/open_jtalk/src/build && cd $TOOLS/open_jtalk/src/build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON .. && make install
#RUN cp $TOOLS/open_jtalk/src/build/*.so* /usr/lib64-nvidia
RUN cd $TOOLS && git clone https://github.com/r9y9/pyopenjtalk.git
RUN cd $TOOLS/pyopenjtalk && pip install .
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/usr-local-lib.conf
RUN ldconfig -v
WORKDIR $PROJECT
RUN $PROJECT/espnet/utils/download_from_google_drive.sh \
    https://drive.google.com/open?id=1OwrUQzAmvjj1x9cDhnZPp6dqtsEqGEJM downloads tar.gz > /dev/null 2>&1
RUN $PROJECT/espnet/utils/download_from_google_drive.sh \
    https://drive.google.com/open?id=1mEnZfBKqA4eT6Bn0eRZuP6lNzL-IL3VD downloads tar.gz > /dev/null 2>&1
EXPOSE 5001
CMD python src/run_server.py
