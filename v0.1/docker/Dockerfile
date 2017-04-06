FROM ubuntu:16.04

RUN apt-get update

# set dependency && install tensorflow
RUN apt-get install -y python3-pip python3-dev git vim && pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp35-cp35m-linux_x86_64.whl

# install pandas package
RUN pip3 install -Iv pandas==0.19.1

# TensorBoard
EXPOSE 6006

WORKDIR /root

COPY .ssh /root/.ssh
# COPY "from host" "to container"
# RUN cp .ssh /root/.ssh
RUN git clone git@github.com:PrioKim/Failure_Prediction

# ENTRY ?? 
CMD ["/bin/bash"]

#RUN ln -s /usr/bin/python3.4 /usr/bin/python
