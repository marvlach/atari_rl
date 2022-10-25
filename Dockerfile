FROM tensorflow/tensorflow:2.10.0

WORKDIR /atari_rl

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && apt-get upgrade -y

# install gym[atari]
RUN pip install gym[atari]==0.26.2

# install open cv
RUN pip install opencv-python==4.6.0

COPY . ./







