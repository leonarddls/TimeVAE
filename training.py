
import tensorflow as tf
from tensorflow.keras.losses import Loss
import numpy as np

def get_losses(x, vae):
    """
    Computes/gets the losses and returns the tensors.
    """
    mse_loss = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    reconstructed = vae(x)
    # Compute reconstruction loss
    flatten_inputs = tf.reshape(x, shape=[-1])
    flatten_outputs = tf.reshape(reconstructed, shape=[-1])

    loss_mse = mse_loss(flatten_inputs, flatten_outputs)# * 12288 # (64x64x3)
    loss_reconstruction = sum(vae.losses)
    loss_total = loss_mse + loss_reconstruction

    return(loss_total, loss_mse, loss_reconstruction)



class VectorMeanSquaredError(Loss):
    def __init__(self):
        super(VectorMeanSquaredError,self).__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float64')
        y_pred = tf.cast(y_pred, 'float64')

        error = y_true - y_pred
        squared_error = tf.math.square(error)
        reduced_sq_er = tf.math.reduce_mean(squared_error, axis= [-2,-1])
        return reduced_sq_er


def get_losses_vect(x, vae):
    """
    Computes/gets the losses and returns the tensors.
    """
    mse_loss = VectorMeanSquaredError()

    reconstructed = vae(x)
    # Fetch reconstruction loss from model
    loss_reconstruction = tf.cast(vae.losses[0] ,tf.float32)  
    # Compute reconstruction loss
    loss_mse = tf.cast(mse_loss(x, reconstructed),tf.float32)
    # Compute total loss
    loss_total = loss_mse + loss_reconstruction
    
    return(loss_total, loss_mse, loss_reconstruction)

def apply_gradient(optimizer,vae,x):
    """
    Gets the gradients on the losses and applies it to the model.
    returns the value of the loss.
    """
    with tf.GradientTape() as tape:
        loss_total, loss_mse, loss_reconstruction = get_losses(x, vae)
    
    grads = tape.gradient(loss_total, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    return(loss_total.numpy(), loss_mse.numpy(), loss_reconstruction.numpy())


def apply_gradient_vect(optimizer,vae,x):
    """
    Gets the gradients on the losses and applies it to the model.
    returns the value of the loss.
    """
    with tf.GradientTape() as tape:
        loss_total, loss_mse, loss_reconstruction = get_losses(x, vae)

        def loss_fn():
            return loss_total

    # grads = tape.gradient(loss_total, vae.trainable_weights)
        grads_and_vars  = optimizer.compute_gradients(loss_fn, vae.trainable_weights, gradient_tape=tape)
    optimizer.apply_gradients(grads_and_vars)

    return(loss_total.numpy().mean(), loss_mse.numpy().mean(), loss_reconstruction.numpy().mean())

def perform_validation(validation_dataset,vae):

    total_losses = []
    mse_losses = []
    reconstruction_losses = []

    for step, batch in enumerate(validation_dataset):

        loss_total, loss_mse, loss_reconstruction = get_losses(batch, vae)

        #convert to numpy
        total_losses.append(loss_total.numpy())
        mse_losses.append(loss_mse.numpy())
        reconstruction_losses.append(loss_reconstruction.numpy())

    mean_total_losses = np.array(total_losses).mean()
    mean_mse_losses = np.array(mse_losses).mean()
    mean_reconstruction_losses = np.array(reconstruction_losses).mean()

    return mean_total_losses, mean_mse_losses, mean_reconstruction_losses


def perform_validation_vect(validation_dataset,vae):

    total_losses = []
    mse_losses = []
    reconstruction_losses = []

    for step, batch in enumerate(validation_dataset):

        loss_total, loss_mse, loss_reconstruction = get_losses_vect(batch, vae)

        #convert to numpy
        total_losses.append(loss_total.numpy().mean())
        mse_losses.append(loss_mse.numpy().mean())
        reconstruction_losses.append(loss_reconstruction.numpy().mean())

    mean_total_losses = np.array(total_losses).mean()
    mean_mse_losses = np.array(mse_losses).mean()
    mean_reconstruction_losses = np.array(reconstruction_losses).mean()

    return mean_total_losses, mean_mse_losses, mean_reconstruction_losses



def train_one_epoch(param,training_dataset,vae):
    """
    Performs one training epochs and collects the loss values
    """
    total_losses = []
    mse_losses = []
    reconstruction_losses = []
    # adapt for 3 losses values
    for step, batch in enumerate(training_dataset):

        if param['private']:
            loss_total, loss_mse, loss_reconstruction = apply_gradient_vect(
                param['optimizer'],
                vae,
                batch)
        else:
            loss_total, loss_mse, loss_reconstruction = apply_gradient(
                param['optimizer'],
                vae,
                batch)
        
        total_losses.append(loss_total)
        mse_losses.append(loss_mse)
        reconstruction_losses.append(loss_reconstruction)

    mean_total_losses = np.array(total_losses).mean()
    mean_mse_losses = np.array(mse_losses).mean()
    mean_reconstruction_losses = np.array(reconstruction_losses).mean()

    return mean_total_losses, mean_mse_losses, mean_reconstruction_losses