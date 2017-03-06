from model import *

class CNN(Model):
  
  ### model parameter ###
  # size of input matrix(two-dimention)
  # fully connected NN nodes(array) e.g. [3,4,5,2]
  # size of filtuer matrix(two-dimention)
  # the number of each size of filter
  
  ### train parameter ###


  ### CV parameter ###

  def __init__(self, num_nodes, num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0):
    pass

  def create_model(self):
    pass
  
  def restore(self):
    pass

  def train(self):
    pass

  def run(self):
    print ("I'm Child")
    pass

  def save(self):
    pass

  def eval(self):
    pass

if __name__ == "__main__":
  cnn = CNN()
  cnn.run()
