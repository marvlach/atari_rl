docker image rm -f atari
docker build -t atari --build-arg http_proxy=http://172.25.1.1:8080 --build-arg https_proxy=http://172.25.1.1:8080 ./