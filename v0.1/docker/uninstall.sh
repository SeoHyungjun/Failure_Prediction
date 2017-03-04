#! /bin/bash

docker rmi fp:dev_env


docker rm $(docker ps -a -q)                    # rm stopped container
docker rmi $(docker images -q -f dangling=true) # rm no named image 
