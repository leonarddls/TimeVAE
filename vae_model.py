import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, Reshape, Conv1DTranspose ,BatchNormalization



def encoder_layers(param, inputs):
    
    x = inputs

    for i, num_filters in enumerate(param['hidden_layer_sizes']):
        x = Conv1D(
                filters = num_filters, 
                kernel_size=3, 
                strides=2, 
                activation='relu', 
                padding='same',
                name=f'enc_conv_{i}')(x)
        if param['batch_norm_enc']:
            x = tf.keras.layers.BatchNormalization()(x)

    x = Flatten(name='enc_flatten')(x)

    
    mu = Dense(param['latent_dim'], name="mu")(x)
    sigma = Dense(param['latent_dim'], name="sigma")(x)

    return mu, sigma

class Sampling(tf.keras.layers.Layer):
  def call(self, inputs):
    """Generates a random sample and combines with the encoder output
    
    Args:
      inputs -- output tensor from the encoder

    Returns:
      `inputs` tensors combined with a random sample
    """

    mu, sigma = inputs
    batch = tf.shape(sigma)[0]
    dim = tf.shape(sigma)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    z = mu + tf.exp(0.5 * sigma) * epsilon

    return  z

def encoder_model(param):
  """Defines the encoder model with the Sampling layer
  Args:
    latent_dim -- dimensionality of the latent space
    input_shape -- shape of the dataset batch

  Returns:
    model -- the encoder model
    conv_shape -- shape of the features before flattening
  """
  inputs = Input(shape=param['dataset_shape'][1:])
  mu, sigma = encoder_layers(param, inputs)
  z = Sampling()((mu, sigma))
  model = tf.keras.Model(inputs=inputs, outputs=[mu, sigma, z], name="encoder")
  model.summary()
  conv_shape = model.layers[-5].output_shape
  return model, conv_shape


def decoder_layers(param, conv_shape, inputs):
    """Defines the decoder layers.
    Args:
    inputs -- output of the encoder 
    conv_shape -- shape of the features before flattening

    Returns:
    tensor containing the decoded output
    """
    x = inputs
    units = conv_shape[1] * conv_shape[2]

    x = Dense(units, name="dec_dense", activation='relu')(x)
    x = Reshape((conv_shape[1], conv_shape[2]), name="decode_reshape")(x)

    for i, num_filters in enumerate(reversed(param['hidden_layer_sizes'][:-1])):
        x = Conv1DTranspose(
            filters = num_filters, 
                kernel_size=3, 
                strides=2, 
                padding='same',
                output_padding = 0,
                activation='relu', 
                name=f'dec_deconv_{i}')(x)
        if param['batch_norm_enc']:
            x = BatchNormalization()(x)

    # last de-convolution
    x = Conv1DTranspose(
            filters = param['dataset_shape'][-1], 
                kernel_size=3, 
                strides=2, 
                padding='same',
                output_padding = None,
                activation='relu', 
                name=f'dec_deconv__{i+1}')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    return x   



def decoder_model(param, conv_shape):
  """Defines the decoder model.
  Args:
    latent_dim -- dimensionality of the latent space
    conv_shape -- shape of the features before flattening

  Returns:
    model -- the decoder model
  """
  inputs = tf.keras.layers.Input(shape=param['latent_dim'],)
  outputs = decoder_layers(param, conv_shape, inputs)
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name="decoder")
  model.summary()
  return model


def vae_model(param, encoder, decoder):
  """Defines the VAE model
  Args:
    encoder -- the encoder model
    decoder -- the decoder model
    input_shape -- shape of the dataset batch

  Returns:
    the complete VAE model
  """
  inputs = tf.keras.layers.Input(shape=param['dataset_shape'][1:])

  # get mu, sigma, and z from the encoder output
  mu, sigma, z = encoder(inputs)

  # get reconstructed output from the encoder
  reconstructed = decoder(z)

  # define the inputs and outputs to the VAE
  model = tf.keras.Model(inputs=inputs, outputs=reconstructed, name="VAE")

  # add the KL loss
  kl_loss = (-0.5 * tf.reduce_mean(sigma - tf.square(mu) - tf.exp(sigma) + 1, axis=1)) * param['reconstruction_wt']

  model.add_loss(kl_loss)
  
  print(sigma)

  model.summary()
  return model


def get_models(param):
  """Returns the encoder, decoder, and vae models"""

  encoder, conv_shape = encoder_model(param)
  decoder = decoder_model(param, conv_shape)
  vae = vae_model(param, encoder, decoder)

  return encoder, decoder, vae