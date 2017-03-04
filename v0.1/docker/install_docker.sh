#! /bin/bash

# install docker 
wget -qO- https://get.docker.com/ | sh

# set docker group to make use of docker without 'sudo'
sudo gpasswd -a $USER docker
sudo systemctl restart docker
newgrp docker
