FROM movesrwth/stormpy:1.7.0
# Mirror of the following Docker container
# FROM movesrwth/stormpy:ci-release
MAINTAINER Thom Badings <thom.badings@ru.nl>


# Build Slurf
#############
RUN mkdir /opt/slurf
WORKDIR /opt/slurf
# Obtain requirements and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Only then install remainder
COPY . .
# Build
RUN python setup.py develop
