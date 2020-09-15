


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy


#RNN class
class RNN:
    def __init__(self, num_steps, input_size, output_size, num_cells_1a, num_cells_2a, num_cells_1b, num_cells_2b, num_cells_1c, num_cells_1d, num_cells_1e, batch_size, start_learning_rate, global_step, decay_steps, end_learning_rate, power_decay, learning_algo, keep_rate_pass, seed_num):
        #Tensorflow inputs
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [batch_size, num_steps, input_size], name='x')
            self.y = tf.placeholder(tf.float32, [batch_size, num_steps, output_size], name='y')
            self.global_step = tf.Variable(global_step, trainable=False)
            self.learn_rate = tf.train.polynomial_decay(start_learning_rate, self.global_step, decay_steps, end_learning_rate, power_decay, name='learning_rate')
            
        #First RNN module of the first layer
        with tf.variable_scope('lstm_1a'):
            self.lstm_1a = tf.nn.rnn_cell.LSTMCell(num_cells_1a, forget_bias=1.0, state_is_tuple=True)
            self.lstm_1a = tf.nn.rnn_cell.DropoutWrapper(self.lstm_1a, output_keep_prob=keep_rate_pass, seed=seed_no)
            self.lstm_init_state_1a = self.lstm_1a.zero_state(batch_size, dtype=tf.float32)
            self.lstm_output_1a, self.lstm_final_state_1a = tf.nn.dynamic_rnn(self.lstm_1a, self.x, initial_state=self.lstm_init_state_1a, time_major=False)
            
        #First RNN module of the second layer
        with tf.variable_scope('GRU_2a'):
            self.gru_2a = tf.nn.rnn_cell.GRUCell(num_cells_2a)
            self.gru_2a = tf.nn.rnn_cell.DropoutWrapper(self.gru_2a, output_keep_prob=keep_rate_pass, seed=seed_no)
            self.gru_init_state_2a = self.gru_2a.zero_state(batch_size, dtype=tf.float32)
            self.gru_output_2a, self.gru_final_state_2a = tf.nn.dynamic_rnn(self.gru_2a, self.lstm_output_1a, initial_state=self.gru_init_state_2a, time_major=False)
            
        #Second RNN module of the first layer
        with tf.variable_scope('GRU_1b'):
            self.gru_1b = tf.nn.rnn_cell.GRUCell(num_cells_1b)
            self.gru_1b = tf.nn.rnn_cell.DropoutWrapper(self.gru_1b, output_keep_prob=keep_rate_pass, seed=seed_no)
            self.gru_init_state_1b = self.gru_1b.zero_state(batch_size, dtype=tf.float32)
            self.gru_output_1b, self.gru_final_state_1b = tf.nn.dynamic_rnn(self.gru_1b, self.x, initial_state=self.gru_init_state_1b, time_major=False)
            
        #second RNN module of the second layer
        with tf.variable_scope('lstm_2b'):
            self.lstm_2b = tf.nn.rnn_cell.LSTMCell(num_cells_2b, forget_bias=1.0, state_is_tuple=True)
            self.lstm_2b = tf.nn.rnn_cell.DropoutWrapper(self.lstm_2b, output_keep_prob=keep_rate_pass, seed=seed_no)
            self.lstm_init_state_2b = self.lstm_2b.zero_state(batch_size, dtype=tf.float32)
            self.lstm_output_2b, self.lstm_final_state_2b = tf.nn.dynamic_rnn(self.lstm_2b, self.gru_output_1b, initial_state=self.lstm_init_state_2b, time_major=False)
            
        #Third RNN module of the first layer
        with tf.variable_scope('lstm_1c'):
            self.lstm_1c = tf.nn.rnn_cell.LSTMCell(num_cells_1c, forget_bias=1.0, state_is_tuple=True)
            self.lstm_1c = tf.nn.rnn_cell.DropoutWrapper(self.lstm_1c, output_keep_prob=keep_rate_pass, seed=seed_no)
            self.lstm_init_state_1c = self.lstm_1c.zero_state(batch_size, dtype=tf.float32)
            self.lstm_output_1c, self.lstm_final_state_1c = tf.nn.dynamic_rnn(self.lstm_1c, self.x, initial_state=self.lstm_init_state_1c, time_major=False)
            
        #Fourth RNN module of the first layer
        with tf.variable_scope('GRU_1d'):
            self.gru_1d = tf.nn.rnn_cell.GRUCell(num_cells_1d)
            self.gru_1d = tf.nn.rnn_cell.DropoutWrapper(self.gru_1d, output_keep_prob=keep_rate_pass, seed=seed_no)
            self.gru_init_state_1d = self.gru_1d.zero_state(batch_size, dtype=tf.float32)
            self.gru_output_1d, self.gru_final_state_1d = tf.nn.dynamic_rnn(self.gru_1d, self.x, initial_state=self.gru_init_state_1d, time_major=False)
            
        #Fifth RNN module of the first layer
        with tf.variable_scope('lstm_1e'):
            self.lstm_1e = tf.nn.rnn_cell.LSTMCell(num_cells_1e, use_peepholes=True, forget_bias=1.0, state_is_tuple=True)
            self.lstm_1e = tf.nn.rnn_cell.DropoutWrapper(self.lstm_1e, output_keep_prob=keep_rate_pass, seed=seed_no)
            self.lstm_init_state_1e = self.lstm_1e.zero_state(batch_size, dtype=tf.float32)
            self.lstm_output_1e, self.lstm_final_state_1e = tf.nn.dynamic_rnn(self.lstm_1e, self.x, initial_state=self.lstm_init_state_1e, time_major=False)
            
        #Full connected hidden layer
        with tf.variable_scope('hidden_layer_3'):
            self.rnn_out_val = tf.reshape(tf.concat([self.gru_output_2a, self.lstm_output_2b, self.lstm_output_1c, self.gru_output_1d, self.lstm_output_1e], 2), [-1, num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e], name='rnnOutVal')
            self.weights_fc3 = tf.get_variable(shape=[num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e, num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e], initializer=tf.random_normal_initializer(mean=0., stddev=1.,), name='weights_hidden_3')
            self.bias_fc3 = tf.get_variable(shape=[num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e], initializer=tf.constant_initializer(0.1), name='bias_hidden_3')
            with tf.name_scope('hidden_regressor_3'):
                self.fc3 = tf.matmul(self.rnn_out_val, self.weights_fc3) + self.bias_fc3
                self.fc3 = tf.nn.dropout(self.fc3, keep_rate_pass, seed=seed_num, name='Dropout_hidden_3')
        
        #Second full connected hidden layer
        with tf.variable_scope('hidden_layer_4'):
            self.weights_fc4 = tf.get_variable(shape=[num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e, int((num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e) / 2)], initializer=tf.random_normal_initializer(mean=0., stddev=1.,), name='weights_hidden_4')
            self.bias_fc4 = tf.get_variable(shape=[int((num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e) / 2)], initializer=tf.constant_initializer(0.1), name='bias_hidden_4')
            with tf.name_scope('hidden_regressor_4'):
                self.fc4 = tf.matmul(self.fc3, self.weights_fc4) + self.bias_fc4
                self.fc4 = tf.nn.dropout(self.fc4, keep_rate_pass, seed=seed_num, name='Dropout_hidden_4')
        
        #Output layer for regression
        with tf.variable_scope('output_layer'):
            self.weights_out = tf.get_variable(shape=[int((num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e) / 2), output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.,), name='weights_output')
            self.bias_out = tf.get_variable(shape=[output_size], initializer=tf.constant_initializer(0.1), name='bias_output')
            with tf.name_scope('output_regressor'):
                self.pred = tf.matmul(self.fc4, self.weights_out) + self.bias_out
        
        #Computation of the losses        
        with tf.name_scope('losses'):
            self.losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(self.pred, [-1], name='reshape_pred')],
                [tf.reshape(self.y, [-1], name='reshape_target')],
                [tf.ones([batch_size * num_steps], dtype=tf.float32, name='weights_loss')],
                softmax_loss_function=lambda labels, logits: tf.square(tf.subtract(labels, logits))
            )
            with tf.name_scope('avg_loss'):
                self.loss = tf.reduce_mean(self.losses)
                tf.summary.scalar('cost', self.loss)
        
        #Training        
        with tf.name_scope('training'):
            if learning_algo == 0:
                self.train_op = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.loss, global_step=self.global_step)
            elif learning_algo == 1:   
                self.train_op = tf.train.AdamOptimizer(start_learning_rate).minimize(self.loss)


#Variables ################################################################################################
starting_point = 0
time_steps = 25
batch_size = 10
repeat_hist = 1
shuffle = 1
input_size = 1
output_size = 1
num_cells_1a = 32
num_cells_2a = 32
num_cells_1b = 32
num_cells_2b = 32
num_cells_1c = 32
num_cells_1d = 32
num_cells_1e = 32
num_epochs = 500
learn_cycles = 10000
start_learn_rate = 0.04
global_step = 0
decay_steps = num_epochs * learn_cycles
end_learning_rate = 0.000001
power_decay = 0.5
learning_algo = 1
rate_keep_drop_out_pass = 0.667
lambda_l2_reg = 0.005
gpu = 0
seed_no = 1
plot_losses = 0
epoch_reduction = 1
epoch_stop = int(num_epochs * 0.9)
cycles_reduced = 10
###########################################################################################################


#Seeding
tf.reset_default_graph()
tf.set_random_seed(seed_no)
np.random.seed(seed_no)

model = RNN(time_steps, input_size, output_size, num_cells_1a, num_cells_2a, num_cells_1b, num_cells_2b, num_cells_1c, num_cells_1d, num_cells_1e, batch_size, start_learn_rate, global_step, decay_steps, end_learning_rate, power_decay, learning_algo, rate_keep_drop_out_pass, seed_no)

#Selecting the running mode
if gpu == 0:
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.InteractiveSession(config=config)
else:
    sess = tf.InteractiveSession()    

init = tf.global_variables_initializer()
sess.run(init)

lstm_state_1a = 0 
gru_state_2a = 0
gru_state_1b = 0
lstm_state_2b = 0
lstm_state_1c = 0
gru_state_1d = 0
lstm_state_1e = 0
tot_loss = []

#Sample data generation
gen = np.ones((batch_size * num_epochs, time_steps + 1))
filler = 0

for i in range(batch_size * num_epochs):
    for j in range(time_steps + 1):
        gen[i, j] = filler
        
        if j != time_steps:
            filler += 1
        else:
            if repeat_hist == 1:
                filler -= (time_steps - 1)

if shuffle == 1:
    np.random.shuffle(gen)

change_epoch_res = []

#Epoch training
for epoch in range(num_epochs):
    
    xPrep = copy.deepcopy(gen[batch_size * epoch:(batch_size * (epoch + 1)), :time_steps])
    yPrep = copy.deepcopy(gen[batch_size * epoch:(batch_size * (epoch + 1)), 1:])
    xx = np.add(np.sin(xPrep), np.cos(xPrep))[:, :, np.newaxis]
    yy = np.add(np.sin(yPrep), np.cos(yPrep))[:, :, np.newaxis]
    starting_point += time_steps * batch_size
    
    if epoch == epoch_reduction:
        learn_cycles = cycles_reduced
    elif epoch == epoch_stop:
        learn_cycles = 1
    
    #Cycle training
    for cycle in range(int(learn_cycles)):
    
        if cycle == 0:
            feed_dict = {model.x: xx, model.y: yy, }
        else:
            feed_dict = {
                model.x: xx,
                model.y: yy,
                model.lstmInitState1a: lstm_state_1a,
                model.gruInitState2a: gru_state_2a,
                model.gruInitState1b: gru_state_1b,
                model.lstmInitState2b: lstm_state_2b,
                model.lstmInitState1c: lstm_state_1c,
                model.gruInitState1d: gru_state_1d,
                model.lstmInitState1e: lstm_state_1e,
            }
    
        _, loss, lstm_state_1a, gru_state_2a, gru_state_1b, lstm_state_2b, lstm_state_1c, gru_state_1d, lstm_state_1e, pred = sess.run(
            [model.train_op, 
           model.loss, 
           model.lstmFinalState1a, 
           model.gruFinalState2a, 
           model.gruFinalState1b,
           model.lstmFinalState2b, 
           model.lstmFinalState1c,
           model.gruFinalState1d,
           model.lstmFinalState1e,
           model.pred], 
            feed_dict=feed_dict
        )
    
        tot_loss.append(loss) 
        
        l2 = lambda_l2_reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("bias" in tf_var.name))
        model.loss += l2
        
        #Plotting the losses
        if plot_losses == 1:
            plt.figure(2)
            plt.title('Losses')
            plt.plot(tot_loss, linewidth=1.0, linestyle="-")
        
        #Printing
        if cycle == 0:
            print('Initial cost is of Epoch', (epoch + 1), 'is', loss)
            change_epoch_res.append(loss)
        elif (cycle + 1) % 10 == 0:
            print('Epoch', (epoch + 1), 'Cycle', (cycle + 1), 'cost is', loss)

sess.close()

print('Epoch change errors are', change_epoch_res)






























