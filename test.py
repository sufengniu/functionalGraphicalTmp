
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

p = 4
M = 5

batch_size = 8
epsilon = 1e-8

def reparameterize(mean, var, is_logvar=True):
    eps = tf.random.normal(shape=var.shape)
    return eps * var + mean if is_logvar is False else eps * tf.exp(var * .5) + mean

def get_symmetric(matrix):
    if len(matrix.shape) == 2:
        shape = [1,0]
    elif len(matrix.shape) == 3:
        shape = [0,2,1]
    else:
        raise Exception("matrix dimension must smaller than 3, [batch_size, dim, dim]")

    upper = tf.linalg.band_part(matrix, 0, -1)
    symm = upper + tf.transpose(upper, shape)
    return symm

def repeat_blockwise(matrix, num):
    # repeat blockwise by num*num
    # input matrix [batch size, n, n]
    # return matrix [batch size, n*num, n*num]
    tiled_matrix = tf.tile(matrix[:,:,:,tf.newaxis,tf.newaxis], [1,1,1,num,num])
    row_wise = tf.concat(tf.unstack(tiled_matrix, axis=1), axis=2)
    col_wise = tf.concat(tf.unstack(row_wise, axis=1), axis=2)
    return col_wise

def decoder(z=None, generate=False):
    # lamb = tf.Variable(tf.random.uniform([p,p], minval=0.0, maxval=1.0), name="lambda")
    lamb = tf.random.uniform([p,p], minval=0.0, maxval=1.0) # lambda to generate prior 
    lamb_symm = get_symmetric(lamb) 

    # off-diagonal lambda 
    offdiagonal_lambda = tf.linalg.set_diag(lamb_symm, tf.zeros(lamb_symm.shape[0:-1]))
    # diagonal lambda
    diagonal_lambda = tf.linalg.diag_part(lamb_symm) * 0.5

    if generate is False:
        mu, logvar, a, b, exp_lambda = z
        # obtain off-diagonal theta
        mu = tf.reshape(repeat_blockwise(tf.reshape(mu, [-1,p,p]), M), [-1,1])
        logvar = tf.reshape(repeat_blockwise(tf.reshape(logvar, [-1,p,p]), M), [-1,1])
        theta_ = tf.reshape(reparameterize(mu, logvar, is_logvar=True), [-1, p*M, p*M])

        symm_theta = get_symmetric(theta_)

        mask = tf.linalg.set_diag(tf.ones([p, p]), tf.zeros(p))
        theta_mask = repeat_blockwise(mask[tf.newaxis,:], M)

        symm_offdiagonal_theta = theta_mask * symm_theta
        
        beta_diagonal = tf.tile(tf.expand_dims(exp_lambda, axis=-1), [1,M])
        theta_diagonal = tf.random.gamma([batch_size], alpha=1, beta=beta_diagonal)
        
        # obtain diagonal theta
        D = tf.zeros_like(theta)
        D = tf.linalg.set_diag(D, tf.reshape(theta_diagonal, [-1,p*M]))

        theta = symm_offdiagonal_theta + D
    else:
        # gamma distribution
        alpha = tf.constant((M**2 + 1) / 2)
        beta = tf.math.square(offdiagonal_lambda) / 2
        tau = tf.random.gamma([batch_size], alpha=alpha, beta=beta)
        tau = tf.linalg.set_diag(tau, tf.zeros(tau.shape[0:-1])) # remove inf 
        tau = get_symmetric(tau)

        # exponential distribution
        beta_diagonal = tf.tile(tf.expand_dims(diagonal_lambda / 2, axis=-1), [1,M])
        theta_diagonal = tf.random.gamma([batch_size], alpha=1, beta=beta_diagonal)

        T = repeat_blockwise(tau, M)
     
        T = reparameterize(mean=tf.constant(0.0), var=tf.reshape(T, [-1, 1]), is_logvar=False)
        theta = tf.reshape(T, [batch_size, p*M, p*M]) # [batch_size, p*M, p*M]

        D = tf.zeros_like(theta)
        D = tf.linalg.set_diag(D, tf.reshape(theta_diagonal, [-1,p*M]))

        theta = theta + D

    # based on theta, generate a postive semi-definite matrix as covariance matrix
    psd_theta = tf.matmul(theta, tf.transpose(theta, [0,2,1]))
    mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros([batch_size, p*M]), covariance_matrix=psd_theta)
    pred = mvn.sample()
    return pred


def encoder(X):
    inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(M*p)), 
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(4*p*p+p), # we need p*p mu, sigma, a, b for normal and gamma distribution, and number of p for exponentional
        ]
    )
    z = inference_net(X)
    mu, logvar, a, b = z[:, 0:p*p], z[:, p*p:2*p*p], tf.nn.relu(z[:, 2*p*p:3*p*p])+epsilon, tf.nn.relu(z[:, 3*p*p:4*p*p])+epsilon
    exp_lambda = tf.nn.relu(z[:, -p:]) + epsilon # add tf.nn.relu to make sure it is greater than 0
    
    return mu, logvar, a, b, exp_lambda



def main():

    z = encoder(X)
    predict = decoder(z, generate=False)


