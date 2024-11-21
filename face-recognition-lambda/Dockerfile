#__copyright__   = "Copyright 2024, VISA Lab"
#__license__     = "MIT"

# Define global args
ARG FUNCTION_DIR="/home/app/"
ARG RUNTIME_VERSION="3.8"

#RUN apt-get update \
#    && apt-get install -y cmake ca-certificates libgl1-mesa-glx
FROM python:${RUNTIME_VERSION}-slim AS python-base
RUN python${RUNTIME_VERSION} -m pip install --upgrade pip
ARG FUNCTION_DIR
RUN mkdir -p ${FUNCTION_DIR}
RUN python${RUNTIME_VERSION} -m pip install awslambdaric --target ${FUNCTION_DIR}

FROM python:${RUNTIME_VERSION}-slim
ARG FUNCTION_DIR
WORKDIR ${FUNCTION_DIR}
COPY --from=python-base ${FUNCTION_DIR} ${FUNCTION_DIR}

# Copy function code
COPY requirements.txt requirements2.txt data.pt face-rec.py ${FUNCTION_DIR}
RUN python -m pip install --no-cache-dir -r ${FUNCTION_DIR}/requirements.txt --target ${FUNCTION_DIR} && \
    python -m pip install --no-cache-dir --no-deps facenet-pytorch==2.5.3 --target ${FUNCTION_DIR}
RUN chmod 755 ${FUNCTION_DIR}/face-rec.py

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "face-rec.handler" ]