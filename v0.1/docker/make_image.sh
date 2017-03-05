#!/bin/bash

PWD="$(pwd)"

if [ "$PWD" != "$HOME" ]
then
  echo "copy '~/.ssh'directory in this directory"
  cp -r ~/.ssh .
fi

docker build -t fp:dev_env .

if [ "$PWD" != "$HOME" ]
then
  echo "remove '.ssh' directory in this directory"
  rm -rf .ssh
fi

docker run -it -p 6006:6006 --name FP_final fp:dev_env
