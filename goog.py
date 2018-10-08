import tensorflow as tf
import tensorflow.contrib.slim as slim
#from tensorflow.layers.python import arg_scope
from tensorflow.contrib.framework import arg_scope

from tensorflow.contrib.framework import add_arg_scope
#n_classes=10
#is_training=True
#learning_rate=0.0001



class googlenet:

  def __init__(self,height,width,n_classes,is_training,learning_rate):
    self.n_classes=n_classes
    self.height=height
    self.width=width
    self.learning_rate=learning_rate
    self.is_training=is_training
    self.image_placeholder = tf.placeholder(tf.float32,shape=[None,self.height,self.width,3],name='input')


  def inception_module(self,inp,info,mod_name):

   with tf.variable_scope(mod_name):
    
    #inp = tf.placeholder(tf.float32,[None,28,28,192])

    #with arg_scope([tf.layers.conv2d],padding='SAME',activation=tf.nn.relu):
   
      conv_0_1 = tf.layers.conv2d(inputs=inp,filters=info['1x1'],kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_1x1')

      conv_0_2 = tf.layers.conv2d(inputs=inp,filters=info['3x3_reduce'],kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_3x3_red')
      conv_1_2 = tf.layers.conv2d(inputs=conv_0_2,filters=info['3x3'],kernel_size=[3,3],padding='SAME',activation=tf.nn.relu,name=mod_name+'_3x3')


      conv_0_3 = tf.layers.conv2d(inputs=inp,filters=info['5x5_reduce'],kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_5x5_red')
      conv_1_3 = tf.layers.conv2d(inputs=conv_0_3,filters=info['5x5'],kernel_size=[5,5],padding='SAME',activation=tf.nn.relu,name=mod_name+'_5x5')

      conv_0_4 = tf.layers.max_pooling2d(inputs=inp,pool_size=[3,3],strides=1,padding='SAME',name=mod_name+'_3x3_maxpool')
      conv_1_4 = tf.layers.conv2d(inputs=conv_0_4,filters=info['pool_proj'],kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_pool_proj')

      output = tf.concat([conv_0_1,conv_1_2,conv_1_3,conv_1_4],axis=3,name=mod_name+'_Filt_concat')
      #print(output.shape)

      return output


  def inception_module_with_aux_classifier(self,inp,info,mod_name):
   training = self.is_training
   if training==None:
     raise ValueError('PLease specify the training Flag')

   with tf.variable_scope(mod_name):
    
      ground_truth_input = tf.placeholder(tf.float32,[None,self.n_classes])

      #with arg_scope([tf.layers.conv2d],padding='SAME',activation=tf.nn.relu):
   
      conv_0_1 = tf.layers.conv2d(inputs=inp,filters=info['1x1'],kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_1x1')

      conv_0_2 = tf.layers.conv2d(inputs=inp,filters=info['3x3_reduce'],kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_3x3_red')
      conv_1_2 = tf.layers.conv2d(inputs=conv_0_2,filters=info['3x3'],kernel_size=[3,3],padding='SAME',activation=tf.nn.relu,name=mod_name+'_3x3')


      conv_0_3 = tf.layers.conv2d(inputs=inp,filters=info['5x5_reduce'],kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_5x5_red')
      conv_1_3 = tf.layers.conv2d(inputs=conv_0_3,filters=info['5x5'],kernel_size=[5,5],padding='SAME',activation=tf.nn.relu,name=mod_name+'_5x5')

      conv_0_4 = tf.layers.max_pooling2d(inputs=inp,pool_size=[3,3],strides=1,padding='SAME',name=mod_name+'_3x3_maxpool')
      conv_1_4 = tf.layers.conv2d(inputs=conv_0_4,filters=info['pool_proj'],kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_pool_proj')

      aux_classifier = tf.layers.average_pooling2d(inp,pool_size=[5,5],strides=3,padding='SAME',name=mod_name+'_aux_avg_pool')
      aux_classsifier = tf.layers.conv2d(aux_classifier,filters=128,kernel_size=[1,1],padding='SAME',activation=tf.nn.relu,name=mod_name+'_aux_1x1')
      aux_classifier = tf.layers.flatten(aux_classifier)
      aux_classifier = tf.layers.dense(aux_classifier,units=1024,activation=tf.nn.relu,name=mod_name+'_aux_dense')
      aux_classifier = tf.layers.dropout(aux_classifier,rate=0.7,training=training,name=mod_name+'_dropout')
      aux_classifier = tf.layers.dense(aux_classifier,units=self.n_classes,name=mod_name+'_logits')
      aux_logits = tf.nn.softmax(aux_classifier,name=mod_name+'_softmax')
      #with tf.name_scope('cross_entropy'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth_input, logits=aux_logits,name=mod_name+'_cross_entropy')
      #with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy,name=mod_name+'_total')

      output = tf.concat([conv_0_1,conv_1_2,conv_1_3,conv_1_4],axis=3,name=mod_name+'_Filt_concat')
      #print(output.shape)

      return output, cross_entropy_mean, ground_truth_input
  

  def build(self):

      with tf.name_scope('Ground_Truth_Placeholder'):
        self.ground_truths = tf.placeholder(tf.float32,shape=[None,self.n_classes])
 

      with tf.name_scope('Google_Net'):

         net = tf.layers.conv2d(
            self.image_placeholder,
            filters=64,
            kernel_size=[7,7],
            strides=2,
            activation=tf.nn.relu,
            name='Conv_1')
         print net.shape

         net = tf.layers.max_pooling2d(net,pool_size=[3,3],strides=2,padding='SAME',name='Max_Pool1')
         print net.shape

         net = tf.layers.conv2d(
            net,
            filters=64,
            kernel_size=[1,1],
            padding='SAME',
            activation=tf.nn.relu,
            name='Conv_2_reduce')

         #print net.shape
         net = tf.layers.conv2d(
            net,
            filters=192,
            kernel_size=[3,3],
            strides=1,
            activation=tf.nn.relu,
            padding='SAME',
            name='Conv_2')
         print net.shape   

         net = tf.layers.max_pooling2d(
            net,
            pool_size=[3,3],
            strides=2,
            padding='SAME',
            name='Max_Pool2')

         print net.shape

         info={'1x1':64,'3x3_reduce':96,'3x3':128,'5x5_reduce':16,'5x5':32,'pool_proj':32}
         net = self.inception_module(net,info,mod_name='MOD_3a')
         print net.shape


         info={'1x1':128,'3x3_reduce':128,'3x3':192,'5x5_reduce':32,'5x5':96,'pool_proj':64}
         net = self.inception_module(net,info,mod_name='MOD_3b')
         print net.shape

         net = tf.layers.max_pooling2d(
            net,
            pool_size=[3,3],
            strides=2,
            padding='SAME',
            name='Max_Pool3')
         print net.shape


         info={'1x1':192,'3x3_reduce':96,'3x3':208,'5x5_reduce':16,'5x5':48,'pool_proj':64}
         net,aux_loss_4a,gt_tensor_4a = self.inception_module_with_aux_classifier(net,info,mod_name='MOD_4a')
         print net.shape

         info={'1x1':160,'3x3_reduce':112,'3x3':224,'5x5_reduce':24,'5x5':64,'pool_proj':64}
         net = self.inception_module(net,info,mod_name='MOD_4b')
         print net.shape

         info={'1x1':128,'3x3_reduce':128,'3x3':256,'5x5_reduce':24,'5x5':64,'pool_proj':64}
         net = self.inception_module(net,info,mod_name='MOD_4c')
         print net.shape


         info={'1x1':112,'3x3_reduce':144,'3x3':288,'5x5_reduce':32,'5x5':64,'pool_proj':64}
         net,aux_loss_4d,gt_tensor_4d = self.inception_module_with_aux_classifier(net,info,mod_name='MOD_4d')
         print net.shape

         info={'1x1':256,'3x3_reduce':160,'3x3':320,'5x5_reduce':32,'5x5':128,'pool_proj':128}
         net = self.inception_module(net,info,mod_name='MOD_4e')
         print net.shape

         net = tf.layers.max_pooling2d(
            net,
            pool_size=[3,3],
            strides=2,
            padding='SAME',
            name='Max_Pool4')
         print net.shape

         info={'1x1':256,'3x3_reduce':160,'3x3':320,'5x5_reduce':32,'5x5':128,'pool_proj':128}
         net = self.inception_module(net,info,mod_name='MOD_5a')
         print net.shape


         info={'1x1':384,'3x3_reduce':192,'3x3':384,'5x5_reduce':48,'5x5':128,'pool_proj':128}
         net = self.inception_module(net,info,mod_name='MOD_5b')
         print net.shape


         net = tf.layers.average_pooling2d(net,pool_size=[7,7],strides=1,padding='VALID',name='Average_Pool1')
         print net.shape

         net = tf.layers.dropout(net,rate=0.4,training=self.is_training,name='Dropout')

         net = tf.layers.flatten(net,name='Flatten')
         print net.shape

         logit = tf.layers.dense(net,units=self.n_classes,activation=None,name='Fully_connected_1')
         print logit.shape


      with tf.name_scope('Final_cross_Entropy'):
          cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ground_truths,logits=logit)
          with tf.name_scope('Total'):
            loss_final = tf.reduce_mean(cross_entropy)#+0.3*aux_loss_4a+0.3*aux_loss_4d)


      with tf.name_scope('train'):
           #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
          optimizer = tf.train.MomentumOptimizer(
           learning_rate=self.learning_rate,
           use_nesterov=True,
           momentum=0.9) #Momentum gradient descent

	
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          with tf.control_dependencies(update_ops):
               tot_loss = loss_final+0.3*aux_loss_4a+0.3*aux_loss_4d 
               train_step = optimizer.minimize(tot_loss)
               tf.summary.scalar('Total-loss',tot_loss)


      #with tf.name_scope('SoftMAx')
      output = tf.nn.softmax(logit,name='SoftMax_output')
      #exit()
      return logit, self.image_placeholder, train_step,self.ground_truths


def main(_):

  print 'making network'
  net = googlenet(height=229,width=229,n_classes=10,is_training=True,learning_rate=0.0001)
  logits, image_placeholder, train_step, gt_placeholder = net.build()

  print 'done making network'
  if not tf.gfile.Exists('./log'):
     tf.gfile.MkDir('./log')

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
     sess.run(init)
     writer = tf.summary.FileWriter('./log/',sess.graph)
     writer.close()


if __name__=='__main__':
   tf.app.run()
   #main()
