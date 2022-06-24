import argparse
from matplotlib.pyplot import get
import tensorflow as tf
import numpy as np
import tensorflow_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from timeit import default_timer as timer

from vae_model import get_models
from data_loading import get_dataset
from training import train_one_epoch, perform_validation_vect, perform_validation  


parser = argparse.ArgumentParser(description="run private timeVAE") 

# private training
parser.add_argument("--private", "-p", type=bool, default=True)

# Dataset parameters
parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--valid_perc', type=float, default=.2)

parser.add_argument('--data_name', '-d', type=str, default='energy')

# Network parameters
parser.add_argument('--latent_dim', type=int, default=20)
parser.add_argument('--hidden_layer_sizes', type=list, default=[200,100])
parser.add_argument('--batch_norm_enc', type=bool, default=False)
parser.add_argument('--batch_norm_dec', type=bool, default=False)
parser.add_argument('--reconstruction_wt', type=float, default=12.)

# Training parameters
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=20)

# DP-SGD parameters
parser.add_argument('--l2_norm_clip', '-l', type=float, default=1.)
parser.add_argument('--noise_multiplier', '-n', type=float, default=.01)
parser.add_argument('--num_microbatches', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=.01)

# get
param =  vars(parser.parse_args())

if param['private']:
    param['optimizer'] =  dp_optimizer.DPAdamGaussianOptimizer(
        l2_norm_clip = param['l2_norm_clip'],
        noise_multiplier = param['noise_multiplier'],
        num_microbatches = param['num_microbatches'] ,
        learning_rate = param['learning_rate'])
else:
    param["optimizer"]  = tf.keras.optimizers.Adam()

print(param)

def main():
    ## Get dataset and scalers
    training_dataset, validation_dataset, scalers = get_dataset(param)
    print('here')
    ## Get VAE model
    encoder, decoder, vae = get_models(param)


    history = {
        'valid_loss':[],
        'valid_loss_reco':[],
        'valid_loss_mse':[],
        'train_loss':[],
        'train_loss_reco':[], 
        'train_loss_mse':[]
        }

    ## Compute privacy score
    

    ## Train the model
    print("Beginning for training")
    for epoch in range(1,param['epochs']+1):
        print('Start of epoch %d' % (epoch,), end='\x1b[1K \r')
        start = timer()

        mean_train_total_losses, mean_train_mse_losses, mean_train_reconstruction_losses = train_one_epoch(param,training_dataset,vae)
        if param['private']:
            mean_val_total_losses, mean_val_mse_losses, mean_val_reconstruction_losses = perform_validation_vect(validation_dataset,vae)
        else:
            mean_val_total_losses, mean_val_mse_losses, mean_val_reconstruction_losses = perform_validation(validation_dataset,vae)

        
        end = timer()

        print(
            f"Epoch {epoch} duration : {np.around(end-start,2)}s, TRAIN total:{round(float(mean_train_total_losses),2)} "+
            f"mse:{round(float(mean_train_mse_losses),2)} reco:{round(float(mean_train_reconstruction_losses),2)}, VALID to"+
            f"tal:{round(float(mean_val_total_losses),2)} mse:{round(float(mean_val_mse_losses),2)} reco:{round(float(mean_val_reconstruction_losses),2)}")
        
        history['valid_loss'].append(mean_val_total_losses)
        history['valid_loss_mse'].append(mean_val_mse_losses)
        history['valid_loss_reco'].append(mean_val_reconstruction_losses)
        history['train_loss'].append(mean_train_total_losses)
        history['train_loss_mse'].append(mean_train_mse_losses)
        history['train_loss_reco'].append(mean_train_reconstruction_losses)


if __name__ == '__main__':
    main()
