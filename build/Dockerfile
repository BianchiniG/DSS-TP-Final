FROM python:3

RUN apt-get update && \
    apt-get install -y unzip wget libgl-dev && \
    wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.zip && \
    unzip cmake-3.19.1.zip && \
    cd cmake-3.19.1 && \
    ./bootstrap && \
    make && \
    make install

EXPOSE 5000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD python app.py