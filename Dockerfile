FROM destination-pred/python

ARG proxy
ENV http_proxy $proxy
ENV https_proxy $proxy

ENV DESTPRED_HOME /app
ENV DESTPRED_DATA /data/

RUN mkdir -p /app && mkdir -p /data/model && mkdir -p /data/cache
COPY src /app
CMD ['cd', '/app']
