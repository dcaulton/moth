ARG BASE_IMAGE=python:3.8.13
FROM ${BASE_IMAGE} as step-base
RUN apt-get -y update && apt-get -y install \
  build-essential 

FROM step-base as gr1
ARG grpath=/frodo
EXPOSE 8002
RUN useradd -m -s /bin/bash -d /home/python python
RUN rm -Rf /mnt/www
WORKDIR $grpath
COPY . $grpath
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
#CMD ["daphne", "-p", "8002", "server.asgi:application"]
