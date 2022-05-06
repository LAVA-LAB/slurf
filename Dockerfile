FROM thombadings/slurf-base:cav22
# Mirror of the following Docker container
# FROM movesrwth/stormpy:ci-release
MAINTAINER Thom Badings <thom.badings@ru.nl>


# Activate virtual environment
############################
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# Build Slurf
#############
RUN mkdir /opt/slurf
WORKDIR /opt/slurf
# Obtain
COPY . .
# Build
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py develop
