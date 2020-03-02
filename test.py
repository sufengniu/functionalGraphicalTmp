
import numpy as np
import tensorflow as tf

p = 4
M = 5

batch_size = 8

def reparameterize(mean, var):
    eps = tf.random.normal(shape=var.shape)
    return eps * var + mean
    # return eps * tf.exp(logvar * .5) + mean

def decoder(z, generate=False):
    # lamb = tf.Variable(tf.random.uniform([p,p], minval=0.0, maxval=1.0), name="lambda")
    lamb = tf.random.uniform([p,p], minval=0.0, maxval=1.0) # lambda to generate prior
    lamb_upper = tf.linalg.band_part(lamb, 0, -1)
    lamb_symm = lamb_upper + tf.transpose(lamb_upper)

    # off-diagonal lambda 
    offdiagonal_lambda = tf.linalg.set_diag(lamb_symm, tf.zeros(lamb_symm.shape[0:-1]))
    # diagonal lambda 
    diagonal_lambda = tf.linalg.diag(tf.linalg.diag_part(lamb_symm)) * 0.5

    # gamma distribution
    alpha = tf.constant((M**2 + 1) / 2)
    beta = tf.math.square(offdiagonal_lambda) / 2
    tau = tf.random.gamma([batch_size], alpha=alpha, beta=beta)
    tau = tf.linalg.set_diag(tau, tf.zeros(tau.shape[0:-1])) # remove inf
    tau = tf.linalg.band_part(tau, 0, -1)
    tau = tau + tf.transpose(tau, [0,2,1])

    # exponential distribution
    beta_diagonal = tf.tile(tf.expand_dims(diagonal_lambda / 2, axis=-1), [1,1,M])
    theta_diagonal = tf.random.gamma([batch_size], alpha=1, beta=beta_diagonal)
    theta_diagonal = tf.where(tf.math.is_inf(theta_diagonal), tf.zeros_like(theta_diagonal), theta_diagonal) # remove inf, [batch_size, p, p, M]

    T = tf.tile(tf.expand_dims(tau, axis=-1), [1,1,1,M*M])

    T = reparameterize(mean=tf.constant(0.0), var=tf.reshape(T, [-1, 1]))
    theta = tf.reshape(T, [batch_size, p, p, M, M]) # [batch_size, p, p, M, M]

    D = tf.reshape(tf.zeros_like(theta), [batch_size, p, p, M, M])
    D = tf.linalg.set_diag(D, theta_diagonal)

    theta = theta + D


def encoder(X):
    