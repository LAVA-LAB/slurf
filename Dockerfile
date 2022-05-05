FROM movesrwth/stormpy:ci-release
MAINTAINER Matthias Volk <m.volk@utwente.nl>


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
