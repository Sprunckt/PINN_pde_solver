import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine import training
from tensorflow.python.ops.gen_control_flow_ops import Exit


class LRSchedule:
  def __init__(self, patience=10, decay_factor=1, min_lr=1e-5, restore_best=False, 
               lr_marks=None, lr_values=None) -> None:
    self.patience = patience
    self.counter = 0
    self.global_counter = 0

    self.lr_ind = 0
    self.lr_marks = lr_marks
    self.lr_values = lr_values

    if self.lr_marks:
      print("using the given learning rate markers")
      assert self.lr_values is not None, "in that mode lr values must be provided"
      self.factor = 1.
      self.nb_marks = len(self.lr_values)
      self._monitor_function = self._update_lr_on_schedule
    else:
      self.factor = decay_factor
      self.nb_marks = 0
      self._monitor_function = self._update_lr_on_val_loss

    self.best_loss = np.inf
    self.trainer = None
    self.min_lr = min_lr
    self.best_weights = None
    self.restore = restore_best
    
  def setTrainer(self, trainer) -> None:
    self.trainer = trainer

  def _update_lr_on_schedule(self, _) -> None:
    self.global_counter += 1
    if self.lr_ind >= self.nb_marks:
      pass
    elif self.global_counter >= self.lr_marks[self.lr_ind]:
        new_lr = self.lr_values[self.lr_ind]
        self.trainer.lr = new_lr
        self.trainer.optimizer.learning_rate = new_lr
        self.lr_ind += 1

  def _update_lr_on_val_loss(self, upgrade) -> None:
    if upgrade:
      self.counter = 0
    else:
      self.counter += 1
      if self.counter >= self.patience:  # update the learning rate
        self.counter = 0
        new_lr = np.maximum(self.factor*self.trainer.lr, 
                            self.min_lr)
        self.trainer.lr = new_lr
        self.trainer.optimizer.learning_rate = new_lr

  def monitor(self):

    if self.restore:
      val_loss = self.trainer.val_history[-1]
      upgrade = val_loss < self.best_loss
      if upgrade:
        self.best_loss = val_loss
        self.best_weights = self.trainer.model.get_weights()

    self._monitor_function(upgrade)

  def train_end(self):
    if self.restore:
      self.trainer.model.set_weights(self.best_weights)

class Solver:

  def __init__(self, val_col, val_bound, val_bound_cond, phys_fun,
                nb_neurons=20, nb_layers=7, lr=1e-2, lr_sch=None, dropout=0.1) -> None:
    """val_col, val_bound are arrays containing the data points"""
    assert val_col.shape[1] == val_bound.shape[1], "collocation and boundary points should have the same dimension"
    assert val_bound_cond.shape[0] == val_bound.shape[0], "invalid number of training points"

    self.input_size = val_col.shape[1]
    self.output_size = val_bound_cond.shape[1]
    self.nb_neurons = nb_neurons
    self.nb_layers = nb_layers

    # defining the model
    inputs = tf.keras.Input(shape=self.input_size)

    outputs = inputs
    for _ in range(self.nb_layers): 
      outputs = self.one_layer(outputs)
      if dropout > 0:
        outputs = tf.keras.layers.Dropout(dropout)(outputs)

    # output layer
    outputs = tf.keras.layers.Dense(self.output_size,  kernel_initializer=tf.keras.initializers.GlorotNormal())(outputs)
    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    self.phys_fun = phys_fun

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    self.lr = lr
    if lr_sch is None:
      self.lr_schedule = LRSchedule()

    else:
      self.lr_schedule = lr_sch

    self.lr_schedule.setTrainer(self)

    self.val_col_var = []
    self.val_bound_var = []
    for i in range(self.input_size):
        self.val_col_var.append(tf.reshape(val_col[:, i], [-1, 1]))
        self.val_bound_var.append(tf.reshape(val_bound[:, i], [-1, 1]))

    self.val_boundary_cond = val_bound_cond
    self.train_history, self.val_history = [], []

  def one_layer(self, inputs, activation=tf.keras.activations.tanh):
    return tf.keras.layers.Dense(self.nb_neurons, activation=activation, 
                                 kernel_initializer=tf.keras.initializers.GlorotNormal())(inputs)

  @tf.function
  def v(self, *args, training=False) -> tf.Tensor:
    v_res = self.model(tf.concat(args, axis=1), training=training)
    return v_res

  def train_step(self, col_var, bound_var, bound_cond):
    mse = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as g:
      physics_res = self.phys_fun(self.v, *col_var, training=True)
      boundary_res = self.v(*bound_var, training=True)
      interior_loss = mse(physics_res, tf.zeros_like(physics_res))
      boundary_loss = mse(boundary_res, bound_cond)
      total_loss = boundary_loss + interior_loss

    # getting the corresponding gradients
    gradients = g.gradient(total_loss, self.model.trainable_variables)

    # applying the gradients
    self.optimizer.apply_gradients(zip(gradients, 
                                       self.model.trainable_variables))

    # saving the losses
    total_loss = total_loss.numpy()
    self.train_history.append(total_loss)

    # computing the validation loss
    physics_res = self.phys_fun(self.v, *self.val_col_var)
    boundary_res = self.v(*self.val_bound_var, training=False)
    interior_loss = mse(physics_res, tf.zeros_like(physics_res))
    boundary_loss = mse(boundary_res, self.val_boundary_cond)
    total_val_loss = (boundary_loss + interior_loss).numpy()
    self.val_history.append(total_val_loss)
    
    # update the lr if necessary
    self.lr_schedule.monitor()
    
    return total_loss, total_val_loss
    
  def train(self, nb_epoch, collocation_points, boundary_points, bound_cond, verbose=True):
    print("starting training")
    assert self.input_size == collocation_points.shape[1] == boundary_points.shape[1], "invalid input dimension"
    assert self.output_size == bound_cond.shape[1]
  	
    train_col_var = []
    train_bound_var = []
    for i in range(self.input_size):
      train_col_var.append(tf.reshape(collocation_points[:, i], [-1, 1]))
      train_bound_var.append(tf.reshape(boundary_points[:, i], [-1, 1]))

    for i in range(nb_epoch):
      train_loss, val_loss = self.train_step(train_col_var, train_bound_var, bound_cond)
      if verbose:
        print("epoch {}/{} --- train_loss : {} --- val_loss : {} -- lr : {}".format(i+1, nb_epoch, 
                                                                                    train_loss, val_loss,
                                                                                    self.lr))
    
    print("training complete")
    self.lr_schedule.train_end()

  def __call__(self, input, training=True):
    return self.model(input, training=training)
