import tensorflow as tf
import numpy as np

from pyDOE import lhs

class LRSchedule:
  def __init__(self, trainer, patience=10, decay_factor=0.2, min_lr=1e-5) -> None:
    self.patience = patience
    self.factor = decay_factor
    self.counter = 0
    self.best_loss = np.inf
    self.trainer = trainer
    self.min_lr = min_lr

  def monitor(self) -> None:
    val_loss = self.trainer.val_history[-1]

    if val_loss < self.best_loss:
      self.counter = 0
      self.best_loss = val_loss
    else:
      self.counter += 1
      if self.counter >= self.patience:  # update the learning rate
        self.counter = 0
        new_lr = np.maximum(self.factor*self.trainer.lr, 
                            self.min_lr)
        self.trainer.optimizer.learning_rate = new_lr


class Solver:

  def __init__(self, val_col, val_bound, val_bound_cond, phys_fun,
                nb_neurons=20, nb_layers=8, lr=1e-2, lr_sch=None) -> None:
    """val_col, val_bound are arrays containing the data points"""
    assert val_col.shape[1] == val_bound.shape[1], "collocation and boundary points should have the same dimension"
    assert val_bound_cond.shape[0] == val_bound.shape[0], "invalid number of training points"

    self.input_size = val_col.shape[1]
    self.output_size = val_bound_cond.shape[1]
    self.nb_neurons = nb_neurons
    self.nb_layers = nb_layers

    inputs = tf.keras.Input(shape=self.input_size)

    outputs = self.one_layer(inputs) 
    for _ in range(1, self.nb_layers - 2): 
      outputs = self.one_layer(outputs)

    # output layer
    outputs = tf.keras.layers.Dense(self.output_size)(outputs)
    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    self.phys_fun = phys_fun

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if lr_sch is None:
      self.lr_schedule = LRSchedule(self)
    else:
      self.lr_schedule = lr_sch

    self.val_col_var = []
    self.val_bound_var = []
    for i in range(self.input_size):
        self.val_col_var.append(tf.reshape(val_col[:, i], [-1, 1]))
        self.val_bound_var.append(tf.reshape(val_bound[:, i], [-1, 1]))

    self.val_boundary_cond = val_bound_cond
    self.train_history, self.val_history = [], []

  def one_layer(self, inputs, activation=tf.keras.activations.tanh):
    return tf.keras.layers.Dense(self.nb_neurons, activation=activation)(inputs)

  @tf.function
  def v(self, *args, training=False) -> tf.Tensor:
    v = self.model(tf.concat(args, axis=1), training=training)
    return v

  def train_step(self, col_var, bound_var, bound_cond):
    mse = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as g:
      physics_res = self.phys_fun(self.v, *col_var)
      boundary_res = self.v(*bound_var)
      interior_loss = mse(physics_res, tf.zeros_like(physics_res))
      boundary_loss = mse(boundary_res, bound_cond)
      total_loss = boundary_loss + interior_loss

    # getting the corresponding gradients
    gradients = g.gradient(total_loss, self.model.trainable_variables)

    # applying the gradients
    self.optimizer.apply_gradients(zip(gradients, 
                                       self.model.trainable_variables))

    # saving the losses
    self.train_history.append(total_loss)

    # computing the validation loss
    physics_res = self.phys_fun(self.v, *self.val_col_var)
    boundary_res = self.v(*self.val_bound_var, training=False)
    interior_loss = mse(physics_res, tf.zeros_like(physics_res))
    boundary_loss = mse(boundary_res, self.val_boundary_cond)
    total_val_loss = boundary_loss + interior_loss
    self.val_history.append(total_val_loss)
    
    # update the lr if necessary
    self.lr_schedule.monitor()
    
    
  def train(self, nb_epoch, collocation_points, boundary_points, bound_cond):
    print("starting training")
    assert self.input_size == collocation_points.shape[1] == boundary_points.shape[1], "invalid input dimension"
    assert self.output_size == bound_cond.shape[1]
  	
    train_col_var = []
    train_bound_var = []
    for i in range(self.input_size):
      train_col_var.append(tf.reshape(collocation_points[:, i], [-1, 1]))
      train_bound_var.append(tf.reshape(boundary_points[:, i], [-1, 1]))

    update_marker = np.floor(0.05 * nb_epoch)
    for i in range(nb_epoch):
      if i > 0 and i % update_marker == 0:
        print(np.round(i * 100 / nb_epoch, 3), "% done")
      self.train_step(train_col_var, train_bound_var, bound_cond)
    
    print("training done")
  def __call__(self, input, training=True):
    return self.model(input, training=training)
