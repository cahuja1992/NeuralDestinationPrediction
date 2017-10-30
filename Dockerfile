FROM destination-pred/python

ARG proxy
ENV http_proxy $proxy
ENV https_proxy $proxy

RUN mkdir -p /app
COPY src /app

