# Use official python base image
FROM python:3

RUN pip install --no-cache-dir scikit_image
RUN useradd cloudknot-user \
    && addgroup cloudknot-user staff \
    && mkdir /home/cloudknot-user \
    && chown -R cloudknot-user:staff /home/cloudknot-user

ENV HOME /home/cloudknot-user
WORKDIR /home/cloudknot-user
RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip
RUN apt-get update && apt-get install zip unzip -y
RUN unzip fiji-linux64.zip
