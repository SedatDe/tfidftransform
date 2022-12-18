FROM python:3.8-slim-buster

WORKDIR /tfidftransform

COPY . .

RUN python setup.py install
