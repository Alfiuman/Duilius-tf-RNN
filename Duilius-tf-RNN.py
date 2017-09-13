



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#RNN class
class RNN:
    def __init__(self, num_steps, input_size, output_size, num_cells_1a, num_cells_2a, num_cells_1b, num_cells_2b, num_cells_1c, num_cells_1d, num_cells_1e, batch_size, start_learning_rate, global_step, decay_steps, end_learning_rate, power_decay, learning_algo, keep_rate_pass, seed_num):
        #Tensorflow inputs
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [batchSize, num_steps, input_size], name='x')
            self.y = tf.placeholder(tf.float32, [batchSize, num_steps, output_size], name='y')
            self.globalStep = tf.Variable(global_step, trainable=False)
            self.learnRate = tf.train.polynomial_decay(start_learning_rate, self.globalStep, decay_steps, end_learning_rate, power_decay, name='learning_rate')
        
        #First RNN module of the first layer
        with tf.variable_scope('lstm_1a'):
            self.lstm1a = tf.nn.rnn_cell.LSTMCell(num_cells_1a, forget_bias=1.0, state_is_tuple=True)
            self.lstm1a = tf.nn.rnn_cell.DropoutWrapper(self.lstm1a, output_keep_prob=keep_rate_pass, seed=seedNo)
            self.lstmInitState1a = self.lstm1a.zero_state(batchSize, dtype=tf.float32)
            self.lstmOutput1a, self.lstmFinalState1a = tf.nn.dynamic_rnn(self.lstm1a, self.x, initial_state=self.lstmInitState1a, time_major=False)
            self.lstmOutput1a = tf.nn.dropout(self.lstmOutput1a, keep_rate_pass, seed=seed_num, name='Dropout_out_1a')
            
        #First RNN module of the second layer
        with tf.variable_scope('GRU_2a'):
            self.gru2a = tf.nn.rnn_cell.GRUCell(num_cells_2a)
            self.gru2a = tf.nn.rnn_cell.DropoutWrapper(self.gru2a, output_keep_prob=keep_rate_pass, seed=seedNo)
            self.gruInitState2a = self.gru2a.zero_state(batchSize, dtype=tf.float32)
            self.gruOutput2a, self.gruFinalState2a = tf.nn.dynamic_rnn(self.gru2a, self.lstmOutput1a, initial_state=self.gruInitState2a, time_major=False)
            self.gruOutput2a = tf.nn.dropout(self.gruOutput2a, keep_rate_pass, seed=seed_num, name='Dropout_out_2a')
            
        #Second RNN module of the first layer
        with tf.variable_scope('GRU_1b'):
            self.gru1b = tf.nn.rnn_cell.GRUCell(num_cells_1b)
            self.gru1b = tf.nn.rnn_cell.DropoutWrapper(self.gru1b, output_keep_prob=keep_rate_pass, seed=seedNo)
            self.gruInitState1b = self.gru1b.zero_state(batchSize, dtype=tf.float32)
            self.gruOutput1b, self.gruFinalState1b = tf.nn.dynamic_rnn(self.gru1b, self.x, initial_state=self.gruInitState1b, time_major=False)
            self.gruOutput1b = tf.nn.dropout(self.gruOutput1b, keep_rate_pass, seed=seed_num, name='Dropout_out_1b')
            
        #second RNN module of the second layer
        with tf.variable_scope('lstm_2b'):
            self.lstm2b = tf.nn.rnn_cell.LSTMCell(num_cells_2b, forget_bias=1.0, state_is_tuple=True)
            self.lstm2b = tf.nn.rnn_cell.DropoutWrapper(self.lstm2b, output_keep_prob=keep_rate_pass, seed=seedNo)
            self.lstmInitState2b = self.lstm2b.zero_state(batchSize, dtype=tf.float32)
            self.lstmOutput2b, self.lstmFinalState2b = tf.nn.dynamic_rnn(self.lstm2b, self.gruOutput1b, initial_state=self.lstmInitState2b, time_major=False)
            self.lstmOutput2b = tf.nn.dropout(self.lstmOutput2b, keep_rate_pass, seed=seed_num, name='Dropout_out_2b')
        
        #Third RNN module of the first layer
        with tf.variable_scope('lstm_1c'):
            self.lstm1c = tf.nn.rnn_cell.LSTMCell(num_cells_1c, forget_bias=1.0, state_is_tuple=True)
            self.lstm1c = tf.nn.rnn_cell.DropoutWrapper(self.lstm1c, output_keep_prob=keep_rate_pass, seed=seedNo)
            self.lstmInitState1c = self.lstm1c.zero_state(batchSize, dtype=tf.float32)
            self.lstmOutput1c, self.lstmFinalState1c = tf.nn.dynamic_rnn(self.lstm1c, self.x, initial_state=self.lstmInitState1c, time_major=False)
            self.lstmOutput1c = tf.nn.dropout(self.lstmOutput1c, keep_rate_pass, seed=seed_num, name='Dropout_out_1c')
            
        #Fourth RNN module of the first layer
        with tf.variable_scope('GRU_1d'):
            self.gru1d = tf.nn.rnn_cell.GRUCell(num_cells_1d)
            self.gru1d = tf.nn.rnn_cell.DropoutWrapper(self.gru1d, output_keep_prob=keep_rate_pass, seed=seedNo)
            self.gruInitState1d = self.gru1d.zero_state(batchSize, dtype=tf.float32)
            self.gruOutput1d, self.gruFinalState1d = tf.nn.dynamic_rnn(self.gru1d, self.x, initial_state=self.gruInitState1d, time_major=False)
            self.gruOutput1d = tf.nn.dropout(self.gruOutput1d, keep_rate_pass, seed=seed_num, name='Dropout_out_1d')
            
        #Fifth RNN module of the first layer
        with tf.variable_scope('lstm_1e'):
            self.lstm1e = tf.nn.rnn_cell.LSTMCell(num_cells_1e, use_peepholes=True, forget_bias=1.0, state_is_tuple=True)
            self.lstm1e = tf.nn.rnn_cell.DropoutWrapper(self.lstm1e, output_keep_prob=keep_rate_pass, seed=seedNo)
            self.lstmInitState1e = self.lstm1e.zero_state(batchSize, dtype=tf.float32)
            self.lstmOutput1e, self.lstmFinalState1e = tf.nn.dynamic_rnn(self.lstm1e, self.x, initial_state=self.lstmInitState1e, time_major=False)
            self.lstmOutput1e = tf.nn.dropout(self.lstmOutput1e, keep_rate_pass, seed=seed_num, name='Dropout_out_1e')
            
        #Full connected hidden layer
        with tf.variable_scope('hidden_layer_3'):
            self.rnnOutVal = tf.reshape(tf.concat([self.gruOutput2a, self.lstmOutput2b, self.lstmOutput1c, self.gruOutput1d, self.lstmOutput1e], 2), [-1, num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e], name='rnnOutVal')
            self.weightsOut = tf.get_variable(shape=[num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e, int((num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e) / 2)], initializer=tf.random_normal_initializer(mean=0., stddev=1.,), name='weights_hidden_3')
            self.biasOut = tf.get_variable(shape=[int((num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e) / 2), ], initializer=tf.constant_initializer(0.1), name='bias_hidden_3')
            with tf.name_scope('hidden_regressor_3'):
                self.hidd = tf.matmul(self.rnnOutVal, self.weightsOut) + self.biasOut
                self.hidd = tf.nn.dropout(self.hidd, keep_rate_pass, seed=seed_num, name='Dropout_hidden_3')
        
        #Output layer for regression
        with tf.variable_scope('output_layer'):
            self.weightsOut = tf.get_variable(shape=[int((num_cells_2a + num_cells_2b + num_cells_1c + num_cells_1d + num_cells_1e) / 2), output_size], initializer=tf.random_normal_initializer(mean=0., stddev=1.,), name='weights_output')
            self.biasOut = tf.get_variable(shape=[output_size, ], initializer=tf.constant_initializer(0.1), name='bias_output')
            with tf.name_scope('output_regressor'):
                self.pred = tf.matmul(self.hidd, self.weightsOut) + self.biasOut
        
        #Computation of the losses        
        with tf.name_scope('losses'):
            self.losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(self.pred, [-1], name='reshape_pred')],
                [tf.reshape(self.y, [-1], name='reshape_target')],
                [tf.ones([batchSize * num_steps], dtype=tf.float32, name='weights_loss')],
                softmax_loss_function=lambda labels, logits: tf.square(tf.subtract(labels, logits))
            )
            with tf.name_scope('avg_loss'):
                self.loss = tf.reduce_mean(self.losses)
                tf.summary.scalar('cost', self.loss)
        
        #Training        
        with tf.name_scope('training'):
            if learning_algo == 0:
                self.train_op = tf.train.GradientDescentOptimizer(self.learnRate).minimize(self.loss, global_step=self.globalStep)
            elif learning_algo == 1:   
                self.train_op = tf.train.AdamOptimizer(start_learning_rate).minimize(self.loss)


#Variables ################################################################################################
startingPoint = 0
timeSteps = 25
batchSize = 100
repeatHist = 1
inputSize = 1
outputSize = 1
numCells1a = 32
numCells2a = 32
numCells1b = 32
numCells2b = 32
numCells1c = 32
numCells1d = 32
numCells1e = 32
numEpochs = 5
learnCycles = 5000
startLearnRate = 0.04
globalStep = 0
decaySteps = numEpochs * learnCycles
endLearningRate = 0.000001
powerDecay = 0.5
learningAlgo = 1
rateKeepDropOutPass = 0.667
gpu = 0
seedNo = 1
plotLosses = 0
test = 0
sizeTest = 2
###########################################################################################################


#Seeding
tf.reset_default_graph()
tf.set_random_seed(seedNo)
np.random.seed(seedNo)

model = RNN(timeSteps, inputSize, outputSize, numCells1a, numCells2a, numCells1b, numCells2b, numCells1c, numCells1d, numCells1e, batchSize, startLearnRate, globalStep, decaySteps, endLearningRate, powerDecay, learningAlgo, rateKeepDropOutPass, seedNo)

#Selecting the running mode
if gpu == 0:
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.InteractiveSession(config=config)
else:
    sess = tf.InteractiveSession()    

init = tf.global_variables_initializer()
sess.run(init)

lstmState1a = 0 
gruState2a = 0
gruState1b = 0
lstmState2b = 0
lstmState1c = 0
gruState1d = 0
lstmState1e = 0
totLoss = []

#Sample data generation
gen = np.ones((batchSize * numEpochs, timeSteps + 1))
filler = 0

for i in range(batchSize * numEpochs):
    for j in range(timeSteps + 1):
        gen[i, j] = filler
        
        if j != timeSteps:
            filler += 1
        else:
            if repeatHist == 1:
                filler -= (timeSteps - 1)
        
np.random.shuffle(gen)

if test != 0:
    useEpochs = numEpochs - sizeTest
else:
    useEpochs = numEpochs

changeEpochRes = []

#Epoch training
for epoch in range(useEpochs):
    
    xPrep = gen[batchSize * epoch:(batchSize * (epoch + 1)), :timeSteps]
    yPrep = gen[batchSize * epoch:(batchSize * (epoch + 1)), 1:]
    xx = np.sin(xPrep)[:, :, np.newaxis]
    yy = np.sin(yPrep)[:, :, np.newaxis]
    startingPoint += timeSteps * batchSize
    
    #Cycle training
    for cycle in range(learnCycles):
    
        if cycle == 0:
            feed_dict = {model.x: xx, model.y: yy, }
        else:
            feed_dict = {
                model.x: xx,
                model.y: yy,
                model.lstmInitState1a: lstmState1a,
                model.gruInitState2a: gruState2a,
                model.gruInitState1b: gruState1b,
                model.lstmInitState2b: lstmState2b,
                model.lstmInitState1c: lstmState1c,
                model.gruInitState1d: gruState1d,
                model.lstmInitState1e: lstmState1e,
            }
    
        _, loss, lstmState1a, gruState2a, gruState1b, lstmState2b, lstmState1c, gruState1d, lstmState1e, pred = sess.run(
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
    
        totLoss.append(loss) 
        
        #Plotting the losses
        if plotLosses == 1:
            plt.figure(2)
            plt.title('Losses')
            plt.plot(totLoss, linewidth=1.0, linestyle="-")
        
        #Printing
        if cycle == 0:
            print('Initial cost is of Epoch', (epoch + 1), 'is', loss)
            changeEpochRes.append(loss)
        elif (cycle + 1) % 10 == 0:
            print('Epoch', (epoch + 1), 'Cycle', (cycle + 1), 'cost is', loss)

print('Epoch change errors are', changeEpochRes)


























