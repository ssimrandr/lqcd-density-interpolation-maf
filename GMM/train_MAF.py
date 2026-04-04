#!/usr/bin/env python

'''This class extends the original `AutoregressiveNetwork` from `tfp.bijectors`,
modifying the output to apply a `tanh` activation to the log-scale. This
modification improves numerical stability by preventing `inf` and `nan` values
during training, and provides more effective regularization of the learned
transformations.

Original implementation: Neumann, M. (2021). PhD Thesis, Bielefeld University.
Available at: https://pub.uni-bielefeld.de/download/2983242/2983243/neumann_PhD_thesis.pdf

Adapted for use with Gaussian Mixture Models (GMM) and (Frankfurt) 5-flavor
 lattice QCD data.'''

import os

#This line is important for keras to interact correctly with TF in the later TF versions !!
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.test as tft
import sys

class Made(tfb.AutoregressiveNetwork):
    def __init__(self, params, event_shape=None, conditional=False, conditional_event_shape=None, conditional_input_layers='first_layer', hidden_units=None,
                 input_order='left-to-right', hidden_degrees='equal', activation=None, use_bias=True,kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001), bias_regularizer=None, kernel_constraint=None, bias_constraint=None, validate_args=False, **kwargs):
        
        super().__init__(params=params, event_shape=event_shape, conditional=conditional, conditional_event_shape=conditional_event_shape,
                         conditional_input_layers=conditional_input_layers, hidden_units=hidden_units, input_order=input_order, hidden_degrees=hidden_degrees,
                         activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                         validate_args=validate_args, **kwargs)
    
    def call(self, x, conditional_input=None):
        

        result = super().call(x, conditional_input=conditional_input)
        
        shift, log_scale = tf.unstack(result, num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)
    
    def get_config(self):
        
        config = super().get_config().copy()
        
        return config
    
    
def compile_MAF_model(num_made, num_inputs, num_cond_inputs=None, made_layers=[128], base_lr=1.0e-3, end_lr=1.0e-5, return_layer_list=False):

  if num_cond_inputs is not None:
    conditional = True
    cond_event_shape = (num_cond_inputs,)
  else:
    conditional = False
    cond_event_shape = None

  made_list = []
  for i in range(num_made):
    made_list.append(tfb.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=Made(params=2, hidden_units=made_layers, event_shape=(num_inputs,), conditional=conditional,
                                    conditional_event_shape=cond_event_shape, activation='relu', name=f"made_{i}"), name=f"maf_{i}"))
    
    made_list.append(tfb.Permute(permutation=np.array(np.arange(0, num_inputs)[::-1])))
                     
  # remove final permute layer
  made_chain = tfb.Chain(list(reversed(made_list[:-1])))

  # we want to transform to gaussian distribution with mean 0.0 and std 1.0 in latent space
  distribution = tfd.TransformedDistribution(
    distribution=tfd.Sample(tfd.Normal(loc=0.0, scale=1.0), sample_shape=[num_inputs]),
    bijector=made_chain)

  x_ = tfk.layers.Input(shape=(num_inputs,), name="aux_input")
  input_list = [x_]

  if conditional:
    c_ = tfk.layers.Input(shape=(num_cond_inputs,), name="cond_input")
    input_list.append(c_)

    current_kwargs = {}
    for i in range(num_made):
      current_kwargs[f"maf_{i}"] = {'conditional_input' : c_}
    
  else:
    current_kwargs = {}
  
  log_prob_ = distribution.log_prob(x_, bijector_kwargs=current_kwargs)
  
  model = tfk.Model(input_list, log_prob_)
  max_epochs = 500  # maximum number of epochs of the training
  learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)
  model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate_fn),
              loss=lambda _, log_prob: -log_prob)
  
  if return_layer_list:
    return model, distribution, made_list
  else:
    return model, distribution


def read_input_file(file_name):
    params = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if '=' in line:
                name, value = line.strip().split('=', 1)
                try:
                    params[name] = float(value)
                except ValueError:
                    params[name] = value 
    return params

#Reading parameters from input file, like the file to read and minmax conditional parameters
input_params = read_input_file('input_train.txt')

num_made = int(input_params.get('num_made', 8))
run_num = 1  # default if not provided

# Override from command-line arguments
# Usage: python train_MAF.py <num_made> <run_num>
if len(sys.argv) > 1:
    num_made = int(sys.argv[1])
if len(sys.argv) > 2:
    run_num = int(sys.argv[2])

csv_file = input_params.pop('csv_file')

vol_min = input_params['a_min']
vol_max = input_params['a_max']
mass_min = input_params['b_min']
mass_max = input_params['b_max']
beta_min = input_params['c_min']
beta_max = input_params['c_max']

def norm_vol(vol):
  return 2 * (vol-vol_min) / (vol_max-vol_min) - 1
def norm_mass(mass):
  return 2 * (mass-mass_min) / (mass_max-mass_min) - 1
def norm_beta(beta):
  return 2 * (beta-beta_min) / (beta_max-beta_min) - 1


data = np.loadtxt(csv_file, delimiter=" ")

chi_min = min(data[:,3])
chi_max = max(data[:,3])
act_min = min(data[:,4])
act_max = max(data[:,4])

def norm_chi(chi):
  return 2 * (chi-chi_min) / (chi_max-chi_min) - 1
def norm_act(act):
  return 2 * (act-act_min) / (act_max-act_min) - 1

def inv_norm_chi(chi_n):
  return chi_min + (chi_max-chi_min)/2*(chi_n+1)
def inv_norm_act(act_n):
  return act_min + (act_max-act_min)/2*(act_n+1)

normed_vols = np.apply_along_axis(norm_vol, 0, data[:,0])
normed_mass = np.apply_along_axis(norm_mass, 0, data[:,1])
normed_betas = np.apply_along_axis(norm_beta, 0, data[:,2])

normed_chis = np.apply_along_axis(norm_chi, 0, data[:,3])
normed_acts = np.apply_along_axis(norm_act, 0, data[:,4])

length = len(normed_vols)

normed_vols = np.reshape(normed_vols, (length,1))
normed_mass = np.reshape(normed_mass, (length,1))
normed_betas = np.reshape(normed_betas, (length,1))

normed_chis = np.reshape(normed_chis, (length,1))
normed_acts = np.reshape(normed_acts, (length,1))

normed_labels = np.concatenate([normed_vols, normed_mass, normed_betas], axis=1)
normed_data = np.concatenate([normed_chis, normed_acts], axis=1)

print(normed_labels)
print(normed_data)
      
save_dir = "/home/ssingh_hpc/BiFra01proj/GMM/trained_models/10k"
os.makedirs(save_dir, exist_ok=True)

#Define one-block made network
epochs = int(input_params['epochs'])
print(num_made)
model, distribution = compile_MAF_model(num_made, num_inputs=2, num_cond_inputs=3)

#as this is once again an unsupervised task, the target vector y is again zeros during training
batch_size = int(input_params['batch_size'])
model.fit(x=[normed_data, normed_labels],
          y=np.zeros((data.shape[0], 0), dtype=np.float32),
          batch_size= batch_size,
          epochs=epochs,
          steps_per_epoch=data.shape[0] // batch_size,
          verbose=1,
          shuffle=True)

model.save_weights(os.path.join(save_dir, f'GMM_made{num_made}_run{run_num}'))
