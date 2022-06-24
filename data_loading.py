"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data !!Not Normalized!!
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)

  data = np.array(data) 
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data. !!Not Normalized!!
  """  
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('datasets/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('datasets/energy_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  print(ori_data.shape)
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return np.array(data)



def get_dataset(param):
  ## Data loading
  # returns shuffle np array dataset
  if param['data_name'] in ['stock', 'energy']:
      full_train_data = real_data_loading(param['data_name'], param['seq_len'])
  elif param['data_name'] == 'sine':
  # Set number of samples and its dimensions
      no, dim = 10000, 6
      full_train_data = sine_data_generation(no, param['seq_len'], dim)
  param['dataset_shape'] = full_train_data.shape
  print('Dataset loaded - shape:', full_train_data.shape)

  ## Train/Test split
  train_data, valid_data =  train_test_split(full_train_data,test_size=param['valid_perc'])
  print("train data shape:", train_data.shape)
  print("validation data shape:",valid_data.shape)

  ## min max scale the data   
  scaled_train_data = np.zeros(shape=train_data.shape)
  scaled_valid_data = np.zeros(shape=valid_data.shape)

  # define scaler
  scalers = []

  # scale training data and save scalers for each dimension
  for i in range(train_data.shape[1]):
      scalers.append(MinMaxScaler())
      scaled_train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :]) 
  # scale validation data
  for i in range(valid_data.shape[1]):
      scaled_valid_data[:, i, :] = scalers[i].transform(valid_data[:, i, :]) 

  ## Convert to TFDS dataset
  # load the training image paths into tensors, create batches and shuffle
  training_dataset = tf.data.Dataset.from_tensor_slices(scaled_train_data)
  training_dataset = training_dataset.batch(param['batch_size'],drop_remainder=True)
  # load the validation image paths into tensors and create batches
  validation_dataset = tf.data.Dataset.from_tensor_slices(scaled_valid_data)
  validation_dataset = validation_dataset.batch(param['batch_size'])
  print(f'number of batches in the training set: {len(training_dataset)}')
  print(f'number of batches in the validation set: {len(validation_dataset)}')

  return training_dataset, validation_dataset, scalers