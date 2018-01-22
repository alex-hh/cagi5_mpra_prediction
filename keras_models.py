"""
The main class here is the CNN class used in all/most my experiments. There is some code for extremely
deep neural networks but I didn't do much with them. The class is used in defining experiments and should
implement a `get_compiled_model` function which returns a compiled model. The class is done such that
the model can be defined without being built and compiled so that its lightweight. Might be worth making
an abstract class/interface. 
"""
from functools import partial, update_wrapper

from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Flatten, Dropout, Dense, Activation, MaxPooling1D,\
                         BatchNormalization, Bidirectional, LSTM, GRU
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.pooling import AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.models import Model, Input, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras import layers
from keras.regularizers import l2
from keras.layers.merge import Concatenate
from keras import metrics
import numpy as np

SEQUENCE_LENGTH = 1000
# from conservation.layers import MixedResPool, CentralFocussedPool
# from conservation.model_utils import load_weights

def batch_apply_func(func, X, batch_size=256):
  start = 0
  end = batch_size
  outputs = []

  while end <= X.shape[0]:
    outputs.append(func([X[start:end]])[0])
    start = end
    end += batch_size
  if X.shape[0] > start:
    outputs.append(func([X[start:X.shape[0]]])[0])
  
  outputs = np.concatenate(outputs, axis=0)
  assert outputs.shape[0] == X.shape[0]
  return outputs

class CNN():
  """
  Convolutional Neural network used throughout most experiments.
  
  args:
      filters - The ith entry gives how many filters/kernels are done in the ith convolutional layer
      kernel_size - The ith entry says how wide the kernels are in the ith convolutional layer
      pooling_size - The ith entry gives the size of the max pooling after the ith convolutional layer
      
      Note - should have len(filters) == len(kernel_size) == len(pooling_size)
      
      dropout - The rate of dropout used in the dense layers
      lr - The learning rate 
      channels - The number of channels in the input data. Defaults to 4 for the A,C,G,T channels
      first_layer_no_bias - If true then no bias term is used in the first convolution, our experiments suggest it 
                            doesn't benefit the network or visualiasation of the first layers
      activation_fn - The activation function values can be relu, lrelu or elu. To add more options the method
                      `get_activation_layer` should be modified accordingly
      dense_units - The length of this array gives the number of hidden fully connected layers (use [] for none) and 
                    each entry gives the number of units in each.
      batch_normalization - Whether to apply batch_normalization. We found it hurt predictions
      reg_weighting - 919 entries which are used to regularize the 919 output neurons with different amounts of 
                      regularization. i.e. hard tasks may be given weighting 1/2 and all others 1
      reg_scaling - To be used with reg_weighting, scales all entries in reg_weighting. 
      pooling_stride - strides in pooling layers; for mixed res pooling layers this should be a list 
                       of (lowres_stride, highres_stride) tuples
  """
  def __init__(self, filters=[64, 64, 64], kernel_size=[15, 15, 15], pooling_size=[8, 2, 2], conv_dropout=0.,
               dense_dropout=0., flatten_dropout=0., lr=0.001, motif_embedding_dim=None,
               channels=4, first_layer_no_bias=False, activation_fn='lrelu', dense_units=[919],
               batch_normalization=False, reg_weighting = None, reg_scaling=1., loss='binary_crossentropy',
               pos_weight=None, output_dim=919, pooling_stride=[8, 2, 2], highres_starts=[None, None, None], 
               highres_ends=[None, None, None], pooltype='max', 
               lowres_pool=16, standardres_pool=12, highres_pool=8):
    
    if len(highres_starts) != len(pooling_stride) and pooltype in ['max', 'central']:
      highres_starts = [None for i in pooling_size]
      highres_ends = highres_starts
    print(kernel_size, len(kernel_size))
    assert len(pooling_stride) == len(pooling_size) == len(kernel_size) == len(filters) == len(highres_starts) == len(highres_ends),\
          "Must specify equal number of pooling and conv paramters, pst {}, ps {}, ks, {}, f {}, hs {} ,he {}".format(len(pooling_stride),
            len(pooling_size), len(kernel_size),len(filters), len(highres_starts), len(highres_ends))
    self.filters = filters
    self.kernel_size = kernel_size
    self.pooling_size = pooling_size
    self.conv_dropout = conv_dropout
    self.dense_dropout = dense_dropout
    self.flatten_dropout = flatten_dropout
    self.lr = lr
    self.channels = channels
    self.first_layer_no_bias = first_layer_no_bias
    self.activation_fn = self.get_activation_layer(activation_fn)
    self.dense_units = dense_units
    self.batch_normalization = batch_normalization
    self.reg_weighting = reg_weighting    
    self.reg_scaling = reg_scaling
    self.model = None
    self.output_dim = output_dim
    self.loss = loss
    if type(pos_weight) == str:
      pos_weight = np.load('/scratch/arh96/conservation/data/{}.npy'.format(pos_weight))
    self.pos_weight = pos_weight
    self.pooling_stride = pooling_stride
    self.highres_starts = highres_starts
    self.highres_ends = highres_ends
    self.motif_embedding_dim = motif_embedding_dim
    self.pooltype = pooltype
    self.lowres_pool = lowres_pool
    self.standardres_pool = standardres_pool
    self.highres_pool = highres_pool

  def get_compiled_model(self):
    loss = self.loss
      # http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
    if self.model == None:
      self.build_model()
    self.model.compile(optimizer=Adam(self.lr), loss=loss, metrics = [metrics.binary_crossentropy])
    return self.model

  def build_model(self, final_layer=True, conv_only=False):
    inputs = Input(shape=(SEQUENCE_LENGTH, self.channels))
    x = self.apply_convolutions(inputs)
    x = Flatten()(x)
    if conv_only:
      self.model = Model(inputs, x)
      return  
    x = Dropout(self.flatten_dropout)(x)
    x = self.apply_dense(x)
    if final_layer:
      x = self.apply_final_layer(x)
    self.model = Model(inputs, x)

  def layer_activations(self, k, X, batch_size=100):
    # todo: make this happen in batches
    # compute activations for the kth layer
    inp = self.model.input
    out = self.model.layers[k].output
    func = K.function([inp], [out])
    return batch_apply_func(func, X, batch_size=batch_size)

  def pooled_layer_activations(self, k, X, batch_size=100):
    inp = self.model.input
    out = self.model.layers[k].output
    out = K.mean(out, axis=1)
    func = K.function([inp], [out])
    return batch_apply_func(func, X, batch_size=batch_size)

  def get_layer_features(self):
    self.layer_features = []

    for i, l in enumerate(self.model.layers):
      if l.__class__.__name__ == 'Conv1D':
        self.layer_features.append((i, l.output_shape[-1]))
      if l.__class__.__name__ == 'Dense':
        self.layer_features.append((i, l.output_shape[-1]))

  def concat_layer_activations(self, X, batch_size=100):
    # TODO - it'd be nice to average pool only over the bits that have the central nucleotide within receptive field
    # I could handle this during model construction
    # TODO batch these
    inp = self.model.input
    outputs = []
    self.layer_features = []

    for i, l in enumerate(self.model.layers):
      if l.__class__.__name__ == 'Conv1D':
        # avg_pool
        outputs.append(K.mean(l.output, axis=1))
        self.layer_features.append((i, l.output_shape[-1]))
      if l.__class__.__name__ == 'Dense':
        outputs.append(l.output)
        self.layer_features.append((i, l.output_shape[-1]))
    out = K.concatenate(outputs, axis=-1)
    func = K.function([inp], [out])
    
    return batch_apply_func(func, X, batch_size=batch_size)

  def apply_convolutions(self, x):
    x_length = 1000
    embed_i = 0 # used to iterate over embedding dims
    # print('here', flush=True)
    for idx, params in enumerate(zip(self.filters, self.kernel_size, self.pooling_size, self.pooling_stride, self.highres_starts, self.highres_ends)):
      f, k, p, p_st, h_start, h_end = params
      if idx == 0 and self.first_layer_no_bias:
        x = Conv1D(f, k, use_bias=False)(x)
        x_length = x_length - k + 1
        # Conv1D input shape (batch_size, steps, input_dim) -- i.e. (batch_size, n_basepairs, input_dim)
        # Conv1D filters are applied to windows of width k of data along the 'steps' dimension (first dimension)
        # All filters span the input_dim dimension (i.e. no dimensionality reduction along input_dim, no strides along input_dim etc.) 
        # Output shape (batch_size, new_steps, f), where f is n_filters
        # without padding (default), new_steps is (steps - k)/stride + 1 (see https://cs231n.github.io/convolutional-networks/)
      else:
        if type(f) in [tuple, list]:
          assert type(k) in [tuple, list] and len(k) == len(f), 'When using filters of multiple sizes, must specify a number of filters for each kernel size'
          # concatenate filters of different sizes
          xs = []
          for fi, ki in zip(f,k):
            xs.append(Conv1D(fi, ki, padding='same')(x))
          # x = K.concatenate(xs, axis=-1)
          x = layers.Concatenate(axis=-1)(xs)
          x_length = x_length
        else:
          x = Conv1D(f, k)(x)
          x_length = x_length - k + 1
        # central_basepair_start = self.central_basepair_starts[-1] -k... something like this. to track receptive fields of central nucleotide
      # print('adding activation', self.activation_fn.__class__.__name__)
      x = LeakyReLU(0.01)(x) # leaky relu has no weights so this should be fine
      if self.batch_normalization:
        x = BatchNormalization()(x)
      # if idx == 0 and p is not None and self.pooltype == 'central':
      #   pool_layer = CentralFocussedPool(layer1_conv=k, lowres_pool=self.lowres_pool, standardres_pool=self.standardres_pool,
      #                                    highres_pool=self.highres_pool)
      #   x = pool_layer(x)
      #   x_length = pool_layer.compute_output_shape((10,x_length,4))[1]
      elif h_start is None and p is not None:     
        x = MaxPooling1D(p, strides=p_st)(x)
        x_length = x_length // p 
      # elif p is not None:
      #   pool_layer = MixedResPool(highres_start=h_start, input_length=x_length,
      #                             highres_end=h_end, pool_length=p, lowres_stride=p_st[0],
      #                             highres_stride=p_st[1])
      #   x = pool_layer(x)
      #   x_length = sum(pool_layer.resolution_lengths())
      if self.conv_dropout > 0 and p is not None:
        x = Dropout(self.conv_dropout)(x)
      if self.motif_embedding_dim:
        # 1x1 convolution projects the motifs onto a smaller space
        # todo - test whether this is better before or after the pool - defo after - think about the dim reduction
        embedding_dim = None
        if idx==0 and type(self.motif_embedding_dim) == int:
          embedding_dim = self.motif_embedding_dim
        elif type(self.motif_embedding_dim)==list:
          if embed_i < len(self.motif_embedding_dim):
            embedding_dim = self.motif_embedding_dim[embed_i]
            embed_i += 1
        if embedding_dim is not None:
          x = Conv1D(embedding_dim, 1, activation=None, use_bias=False)(x)
    return x

  def apply_dense(self, x):
    for units in self.dense_units:
      x = Dense(units)(x)
      x = self.activation_fn(x)
      if self.batch_normalization:
        x = BatchNormalization()(x)
      x = Dropout(self.dense_dropout)(x)
    return x
    
  def apply_final_layer(self, x):
    if self.reg_weighting is not None:
      outputs = []
      for i in range(self.output_dim):
        outputs.append(Dense(1, activation='sigmoid', kernel_regularizer = l2(self.reg_scaling*self.reg_weighting[i]))(x))
      output = Concatenate()(outputs)
    else:
      output = Dense(self.output_dim, activation=None)(x)
      output = Activation('sigmoid')(output)
    return output 

  def get_activation_layer(self, activation_name):
    if activation_name == 'lrelu':
      return LeakyReLU(0.01)
    elif activation_name == 'elu':
      return Activation('elu')
    else:
      return Activation('relu')
