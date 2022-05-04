FROM movesrwth/carl:master14
MAINTAINER Matthias Volk <m.volk@utwente.nl>

# If the necessary changes are incorporated into the main repos of Storm and stormpy,
# the following build steps (up to "Build Slurf") are not necessary anymore.
# Instead, one can use the following base image:
# FROM movesrwth/stormpy:ci-release


# Build Storm
#############
WORKDIR /opt
# Obtain
RUN git clone https://github.com/volkm/storm.git
RUN mkdir -p /opt/storm/build
WORKDIR /opt/storm/build
#RUN git checkout tags/1.6.4
# Configure
RUN cmake .. "-DCMAKE_BUILD_TYPE=Release -DSTORM_DEVELOPER=OFF -DSTORM_LOG_DISABLE_DEBUG=ON $CMAKE_ARGS DSTORM_PORTABLE=ON"
# Build
RUN make binaries -j 1
ENV PATH="/opt/storm/build/bin:$PATH"


# Install dependencies
######################
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    uuid-dev \
    python3 \
    virtualenv


# Create virtual environment
############################
WORKDIR /opt
ENV VIRTUAL_ENV=/opt/venv
RUN virtualenv -p python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# Build pycarl
##############
WORKDIR /opt
# Obtain
RUN git clone https://github.com/moves-rwth/pycarl.git
WORKDIR /opt/pycarl
RUN git checkout tags/2.0.5
# Build
RUN python setup.py build_ext -j 1 develop


# Build stormpy
###############
WORKDIR /opt
# Obtain
RUN git clone https://github.com/volkm/stormpy.git
WORKDIR /opt/stormpy
#RUN git checkout tags/1.6.4
# Build
RUN python setup.py build_ext --storm-dir /opt/storm/build/ -j 1 develop


# Build Slurf
#############
RUN mkdir /opt/slurf
WORKDIR /opt/slurf
# Obtain
COPY . .
# Build
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py develop
