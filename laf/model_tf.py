from tensorflow.keras.layers import *

import tensorflow as tf

class LAFLayerTF(Layer):
    def __init__(self, units=32, eps=1e-7):
        super(LAFLayerTF, self).__init__()
        self.units = units
        self.eps = eps
        
    def build(self, input_shape, kernel_initializer='random_uniform'):
        self.w = self.add_weight(shape=(12, self.units),
                                 initializer=kernel_initializer,
                                 trainable=True)
        self.e = tf.constant([1,-1,1,-1],dtype=tf.float32)
        self.num_idx = tf.constant([1,1,0,0],dtype=tf.float32)
        self.den_idx = tf.constant([0,0,1,1],dtype=tf.float32)
        self.num_idx = tf.reshape(self.num_idx, [1,1,-1,1])
        self.den_idx = tf.reshape(self.den_idx, [1,1,-1,1])
       
    def call(self, inputs):
        inputs, index = inputs
        index = tf.reshape(index,[-1])
        eps = self.eps
        sup = 1.0 - eps
     
        x = tf.clip_by_value(inputs, eps, sup)
        x = tf.expand_dims(x, axis=-1)
        e = tf.reshape(self.e, [1,1,-1])
  
        exps = (1.- e)/2. + x*e
        exps = tf.expand_dims(exps, axis=-1)
        exps = tf.math.pow(exps, tf.nn.relu(  tf.gather(self.w, [1,3,5,7])  ))
      
        scatter = tf.math.segment_sum(exps, index)
        scatter = tf.math.maximum(scatter, eps)
      
        sqrt = tf.math.pow(scatter, tf.nn.relu(  tf.gather(self.w, [0,2,4,6])  ))
        alpha_beta = tf.reshape(tf.gather(self.w, [8,9,10,11]), [1,1,4,-1])
        terms = sqrt * alpha_beta

        num = tf.math.reduce_sum(terms * self.num_idx, axis=2)
        den = tf.math.reduce_sum(terms * self.den_idx, axis=2)

        multiplier = 2.0 * tf.nn.relu(tf.math.sign(den)) - 1.0
      
        den = tf.where( (den < eps) & (den > -eps), multiplier*eps, den)
        res = num / den
        return res

class LAFLayerFastTF(Layer):
    def __init__(self, units=32, eps=1e-7):
        super(LAFLayerFastTF, self).__init__()
        self.units = units
        self.eps = eps
        
    def build(self, input_shape, kernel_initializer='random_uniform'):
        self.w = self.add_weight(shape=(12, self.units),
                                 initializer=kernel_initializer,
                                 trainable=True)
        self.e = tf.constant([1,-1,1,-1],dtype=tf.float32)
        self.num_idx = tf.constant([1,1,0,0],dtype=tf.float32)
        self.den_idx = tf.constant([0,0,1,1],dtype=tf.float32)
        self.num_idx = tf.reshape(self.num_idx, [1,1,-1,1])
        self.den_idx = tf.reshape(self.den_idx, [1,1,-1,1])
       
    def call(self, inputs):
        # Input here is batch_size, seq_length, ...
        eps = self.eps
        sup = 1.0 - eps
     
        x = tf.clip_by_value(inputs, eps, sup)
        x = tf.expand_dims(x, axis=-1)
        e = tf.reshape(self.e, [1,1,-1])
  
        exps = (1.- e)/2. + x*e
        exps = tf.expand_dims(exps, axis=-1)
        exps = tf.math.pow(exps, tf.nn.relu(  tf.gather(self.w, [1,3,5,7])  ))
        
        # exps has dim batch_size, seq_length, ... , 4 
        scatter = tf.reduce_sum(exps, axis=1)
        scatter = tf.math.maximum(scatter, eps)
      
        sqrt = tf.math.pow(scatter, tf.nn.relu(  tf.gather(self.w, [0,2,4,6])  ))
        alpha_beta = tf.reshape(tf.gather(self.w, [8,9,10,11]), [1,1,4,-1])
        terms = sqrt * alpha_beta

        num = tf.math.reduce_sum(terms * self.num_idx, axis=2)
        den = tf.math.reduce_sum(terms * self.den_idx, axis=2)

        multiplier = 2.0 * tf.nn.relu(tf.math.sign(den)) - 1.0
      
        den = tf.where( (den < eps) & (den > -eps), multiplier*eps, den)
        res = num / den
        return res
