#!/bin/bash

cp Dockerfile ~
cp make_image.sh ~
cd ~
docker build -t fp:dev_env .
rm -f Dockerfile
rm -f make_image.sh
docker run -it -p 6006:6006 --name FP_final fp:dev_env
