FROM python:3.12.4-slim-bullseye

WORKDIR /app
ENV CPPFLAGS "-I/usr/include/suitesparse"
ENV CVXOPT_BUILD_GLPK 1

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y wget build-essential libopenblas-dev libatlas-base-dev liblapack-dev libsuitesparse-dev libdsdp-dev libfftw3-dev libglpk-dev libgsl-dev

RUN pip install -r requirements.txt

COPY . /app

RUN chmod +x *.sh

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["/bin/bash", "/app/run.sh"]