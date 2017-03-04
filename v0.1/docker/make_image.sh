#!/bin/bash

docker build -t fp:dev_env .
docker run -it -p 6006:6006 --name FP_final fp:dev_env
