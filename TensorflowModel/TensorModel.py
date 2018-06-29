import numpy as np 
import tensoflow as tf



    
def build_model(input_data, parameters):

    with tf.variable_scope('model')    
        layers  = []
        for i, filters in ennumerate(layers):
            with tf.variable_scope('conv{}'.format(i+1))
                out  = tf.layers.conv2D(out, filter = filters, strides = 1, padding = 'same'  )
                out  = tf.nn.relu(out)
                out  = tf.layers.max_pooling2D(out, pool_size = [2,2] ,strides = 2 )

        out = tf.reshape()
        with tf.variable_scope('fc1'):
            out = tf.layers.dense(out , )
            out = tf.nn.relu(out)
        with tf.variable_scope('fc2'):
            logits = tf.layers.dense(out, parameters.num_labels)
    return logits

def model_operation(input, parameters, num_epocs):

    input_data = input['input_data']
    
    logits = build_model(input_data, parameters)
    predictions = tf.reduce_max(logits, -1)

    loss = tf.losses.softmax_coss_entropy(onehot_labels = labels, logits = logists)

    leaarning_rate = parameters['learning_rate'] 
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    with tf.variable_scope('metrics'):
        metrics = {
            'loss' : tf.reduce_mean(loss)
            'accuracy': tf.metrics.accuracy(labels = input['labels'] , predictions = predictions)
        }

    model_spec['initialize_variables'] = tf.global_variables_initializer()

    model_spec['logits'] = logits
    model_spec['loss'] = loss
    model_spec['optimizer'] = optimizer
    model_spec['metrics'] = metrics

    return model_spec