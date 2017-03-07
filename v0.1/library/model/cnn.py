from model import *

class CNN(Model):
 
  ### CV parameter ###

  def __init__(self):
  ### env parameter(init) ###
    pass


  def create_model(self, input_size, num_NN_nodes, filter_sizes, num_filters, dropout_keep_prob=1.0, l2_reg_lambda=0.0):
  ### model parameter(create_model) ###
  # input_size : size of input matrix(two-dimention) e.g.[3,4]
  # num_NN_nodes : fully connected NN nodes(array) e.g. [3,4,5,2]
  # filter_sizes : size of filter matrix(two-dimention)
  # numb_filters : the number of each size of filter
  # regularization : dropout_keep_prob, l2_reg_lambda(when not applied, each value are 1.0, 0.0)
  # =============================== ###
    pass



  def restore(self):
    pass

  def save(self):
    print ("Save model trained!!!")
    pass

  def eval(self):
    print ("Eval model trained!!!")
    pass



  def train(self):
  ### train parameter ###
  # dev_sample_percentage : percentage of the training data to use for validation"
  # data_file_location : Data source for training
  # out_subdir : directory for saving output
  # tag : added in output directory name
  # batch_size : Batch Size
  # num_epochsNumber of training epochs
  # train_limit : train limit when there are no improvemnt in several vailidation steps. using as 'train_limit*evalutate_every', means step size limit
  # checkpoint_every : Save model after this many steps (default: 150)
  # allow_soft_placement : Allow device soft device placement
  # log_device_placement : Log placement of ops on devices
  # =============================== ###  
    pass


  def run(self):
    print ("Predict Something!!!!")
    pass




if __name__ == "__main__":
  cnn = CNN()
  cnn.run()
