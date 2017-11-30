import tensorflow as tf


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
            num_input_channels,
            conv_filter_size,
            num_filters,
            activation,
            pooling,
            win_strd_size):
    weights=create_weights(shape=[conv_filter_size,
                conv_filter_size,num_input_channels,num_filters])
    biases=create_biases(num_filters)
    layer=tf.nn.conv2d(input=input,
                filter=weights,
                strides=[1,1,1,1],
                padding='SAME')
    layer=tf.nn.bias_add(layer,biases)
    #if activation!="no":
        #layer=eval("tf.nn."+activation+"(layer)")
    layer=eval("tf.nn."+pooling+"(layer,ksize=[1,win_strd_size,win_strd_size,1], \
                   strides=[1,win_strd_size,win_strd_size,1],padding='SAME')")
    if activation!="no":
        layer=eval("tf.nn."+activation+"(layer)")
    return layer

def create_flatten_layer(layer):
    layer_shape=layer.get_shape()
    num_features=layer_shape[1:4].num_elements()
    layer=tf.reshape(layer,[-1,num_features])
    return layer

def create_fc_layer(input,num_inputs,num_outputs,dropout,activation):
    weights=create_weights(shape=[num_inputs,num_outputs])
    biases=create_biases(num_outputs)
    layer=input
    if dropout!="no":
        layer=tf.nn.dropout(layer,0.5)
    layer=tf.matmul(layer,weights)+biases
    if activation!="no":
        layer=eval("tf.nn."+activation+"(layer)")
    return layer
 
         
       
    
 
