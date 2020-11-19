ARG BASE_IMAGE="ubuntu:latest"
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    python3-dev \
    python3-distutils \
    python3-venv \
    openjdk-11-jre-headless \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

RUN python3 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN pip install -U pip setuptools

RUN pip install --no-cache-dir torch torchvision torchserve

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY config.properties /home/model-server/config.properties
COPY model-store /home/model-server/model-store

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server \
    && chown -R model-server /home/venv

RUN mkdir /home/model-server/model-store \
    && chown -R model-server /home/model-server/model-store \
    && cd /home/model-server/model-store

EXPOSE 8080 8081 8082

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]