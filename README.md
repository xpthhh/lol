account.github.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loading import load_mnist
from model_CNN import CNN

%matplotlib inline


" Data Loading "
data = load_mnist()
x_train, x_test, y_train, y_test, y_train_cls, y_test_cls = data


" Model Construction "
tf.reset_default_graph()
g = tf.Graph() 
with g.as_default():
    with tf.Session(graph=g) as sess:
        with tf.variable_scope("cnn_mnist"):
            classifier = CNN(sess, data)
            classifier.graph_construction()
            tf.global_variables_initializer().run(session=sess)
            classifier.train()
            classifier.save(sess=sess, save_path='graphs/cnn_mnist')
            classifier.print_test_accuracy()
            classifier.print_confusion_matrix()
            classifier.plot_9_test_images_with_false_prediction()
            classifier.plot_test_images_in_input_conv1_conv2_layers(num_test_images=2)


" Restore "
tf.reset_default_graph()
g = tf.Graph() 
with g.as_default():
    with tf.Session(graph=g) as sess:
        saver = tf.train.import_meta_graph('graphs/cnn_mnist' + '.meta',
                                           clear_devices=True)
        saver.restore(sess=sess, save_path='graphs/cnn_mnist')

        x         = sess.graph.get_tensor_by_name("cnn_mnist" +'/'+ 'x:0')
        keep_prob = sess.graph.get_tensor_by_name("cnn_mnist" +'/'+ 'keep_prob:0')
        is_train  = sess.graph.get_tensor_by_name("cnn_mnist" +'/'+ 'is_train:0')
        y_pred    = sess.graph.get_tensor_by_name("cnn_mnist" +'/'+ 'y_pred:0')
        
        feed_dict = {x: x_test, keep_prob: 1.0, is_train: False}
        probs = sess.run(y_pred, feed_dict=feed_dict)
        cls_pred = np.argmax(probs, axis=1)
        print("Accuracy on test-set: {:%}".format((cls_pred == y_test_cls).sum() / len(y_test_cls)))
/Users/sungchul/anaconda/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.6 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.5
  return f(*args, **kwds)
/Users/sungchul/anaconda/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters


0 4.5664244
100 0.13108894
200 0.14024976
300 0.09715206
400 0.12203499
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:12
0 0.043305114
100 0.040103085
200 0.060545012
300 0.049387302
400 0.0705938
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:11
Graph Saved
Accuracy on test-set: 98.7%
[[ 978    0    0    0    0    0    0    1    1    0]
 [   0 1133    1    1    0    0    0    0    0    0]
 [   0    2 1028    0    0    0    1    1    0    0]
 [   0    0    1 1004    0    2    0    2    1    0]
 [   0    0    2    0  971    0    4    0    3    2]
 [   2    0    0    5    1  881    1    1    0    1]
 [   7    3    0    0    1    5  942    0    0    0]
 [   1    5   21    2    1    0    0  987    1   10]
 [   4    1    6    2    0    2    1    2  953    3]
 [   2    3    2    1    6    2    0    2    3  988]]

<Figure size 432x288 with 0 Axes>


<Figure size 432x288 with 0 Axes>

<Figure size 432x288 with 0 Axes>


<Figure size 432x288 with 0 Axes>

<Figure size 432x288 with 0 Axes>

INFO:tensorflow:Restoring parameters from graphs/cnn_mnist


Restoring parameters from graphs/cnn_mnist


Accuracy on test-set: 98.650000%
[Back to top]
2 Ensemble of many CNNs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import utils
from data_loading import load_mnist
from model_CNN import CNN

%matplotlib inline


" Data Loading "
data = load_mnist()
x_train, x_test, y_train, y_test, y_train_cls, y_test_cls = data

# data dimension
img_size = 28
img_size_flat = 784
img_shape = (28, 28)
num_classes = 10
num_features = 1764

# number of models
num_models = 10


" Model Construction "
t = time.time()
classifiers = []
test_probs_cumulative = np.zeros((len(y_test_cls),10))
for i in range(num_models):
    tf.reset_default_graph()
    g = tf.Graph() 
    with g.as_default():
        with tf.Session(graph=g) as sess:
            with tf.variable_scope("cnn_mnist_{}".format(str(i))):
                classifiers.append(CNN(sess, data))
    
                classifiers[i].graph_construction()
                tf.global_variables_initializer().run(session=sess)
                classifiers[i].train()
                classifiers[i].save(sess=sess, save_path='graphs/cnn_mnist_{}'.format(str(i)))
                classifiers[i].print_test_accuracy()

                test_probs_cumulative += classifiers[i].get_test_probs()
print("Model Constructed in {} seconds".format(time.time()-t))


" Ensemble of Models - Test Performance "
test_probs_ensemble = test_probs_cumulative / num_models
test_cls_pred_ensemble = np.argmax(test_probs_ensemble, axis=1)
test_accuracy_ensemble = (y_test_cls == test_cls_pred_ensemble).sum() / len(y_test_cls)
print("Accuracy on test-set: {0:.1%}".format(test_accuracy_ensemble))


" Ensemble Prediction - One Image "
image0 = x_test[0]
cls0 = y_test_cls[0]
probs_cumulative = np.zeros((1,10))
for i in range(num_models):
    tf.reset_default_graph()
    g = tf.Graph() 
    with g.as_default():
        with tf.Session(graph=g) as sess:
            saver = tf.train.import_meta_graph('graphs/cnn_mnist_{}'.format(str(i)) + '.meta',
                                               clear_devices=True)
            saver.restore(sess=sess, save_path='graphs/cnn_mnist_{}'.format(str(i)))

            x         = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'x:0')
            keep_prob = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'keep_prob:0')
            is_train  = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'is_train:0')
            y_pred    = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'y_pred:0')

            feed_dict = {x: [image0], keep_prob: 1.0, is_train: False}
            probs_cumulative += sess.run(y_pred, feed_dict=feed_dict)
        
probs_ensemble = probs_cumulative / num_models
cls_pred_ensemble = np.argmax(probs_ensemble, axis=1)
utils.plot_one_image(image0, img_shape=img_shape, cls_true=cls0, cls_pred=cls_pred_ensemble[0])


" Ensemble Prediction - Many Images "
image0 = x_test[0:9]
cls0 = y_test_cls[0:9]
probs_cumulative = np.zeros((9,10))
for i in range(num_models):
    tf.reset_default_graph()
    g = tf.Graph() 
    with g.as_default():
        with tf.Session(graph=g) as sess:
            saver = tf.train.import_meta_graph('graphs/cnn_mnist_{}'.format(str(i)) + '.meta',
                                               clear_devices=True)
            saver.restore(sess, 'graphs/cnn_mnist_{}'.format(str(i)))

            x         = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'x:0')
            keep_prob = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'keep_prob:0')
            is_train  = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'is_train:0')
            y_pred    = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'y_pred:0')

            feed_dict = {x: image0, keep_prob: 1.0, is_train: False}
            probs_cumulative += sess.run(y_pred, feed_dict=feed_dict)
        
probs_ensemble = probs_cumulative / num_models
cls_pred_ensemble = np.argmax(probs_ensemble, axis=1)
utils.plot_many_images_2d(image0, img_shape=img_shape, cls_true=cls0, cls_pred=cls_pred_ensemble)


" Ensemble Prediction - Many Images with Wrong Predictions "
image0 = x_test
cls0 = y_test_cls
probs_cumulative = np.zeros((len(y_test_cls),10))
for i in range(num_models):
    tf.reset_default_graph()
    g = tf.Graph() 
    with g.as_default():
        with tf.Session(graph=g) as sess:
            saver = tf.train.import_meta_graph('graphs/cnn_mnist_{}'.format(str(i)) + '.meta',
                                               clear_devices=True)
            saver.restore(sess, 'graphs/cnn_mnist_{}'.format(str(i)))

            x         = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'x:0')
            keep_prob = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'keep_prob:0')
            is_train  = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'is_train:0')
            y_pred    = sess.graph.get_tensor_by_name("cnn_mnist_{}".format(str(i)) +'/'+ 'y_pred:0')

            feed_dict = {x: image0, keep_prob: 1.0, is_train: False}
            probs_cumulative += sess.run(y_pred, feed_dict=feed_dict)

probs_ensemble = probs_cumulative / num_models
cls_pred_ensemble = np.argmax(probs_ensemble, axis=1)

images_false_prediction = []
cls_true = []
cls_pred = []
num_false_prediction = 0
i = 0
while num_false_prediction < 9:
    if cls_pred_ensemble[i] != cls0[i]: 
        images_false_prediction.append(image0[i])
        cls_true.append(cls0[i])
        cls_pred.append(cls_pred_ensemble[i])
        num_false_prediction += 1
    i += 1

utils.plot_many_images_2d(images=images_false_prediction,
                          img_shape=img_shape,
                          cls_true=cls_true,
                          cls_pred=cls_pred)
0 6.432804
100 0.15222183
200 0.09918217
300 0.12246569
400 0.14798102
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:12
0 0.045931447
100 0.036280293
200 0.041577674
300 0.06410316
400 0.10277444
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:13
Graph Saved
Accuracy on test-set: 98.7%
0 5.745804
100 0.17318669
200 0.078094035
300 0.1035357
400 0.120640375
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:13
0 0.034553476
100 0.07556909
200 0.05722572
300 0.053695295
400 0.04046185
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:11
Graph Saved
Accuracy on test-set: 98.8%
0 4.885208
100 0.13996993
200 0.12196465
300 0.084302396
400 0.10590769
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:07
0 0.0353338
100 0.03794836
200 0.046695646
300 0.115138285
400 0.06760834
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:08
Graph Saved
Accuracy on test-set: 98.8%
0 5.434952
100 0.11948712
200 0.1357146
300 0.11295833
400 0.08610912
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:12
0 0.06452591
100 0.024217378
200 0.06040974
300 0.06444344
400 0.06871896
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:14
Graph Saved
Accuracy on test-set: 98.8%
0 5.778116
100 0.13930745
200 0.075719744
300 0.12984587
400 0.15916938
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.984375
Time Usage: 0:01:14
0 0.04333747
100 0.038528495
200 0.032216407
300 0.05829751
400 0.10621403
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:08
Graph Saved
Accuracy on test-set: 98.8%
0 6.9978657
100 0.18261743
200 0.08967433
300 0.09399176
400 0.20000494
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:09
0 0.062812716
100 0.027170118
200 0.0615841
300 0.069861956
400 0.07009582
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:07
Graph Saved
Accuracy on test-set: 98.8%
0 4.727253
100 0.10179028
200 0.074187845
300 0.08549748
400 0.118408315
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:06
0 0.044832554
100 0.012051318
200 0.05335964
300 0.09262896
400 0.06310369
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:08
Graph Saved
Accuracy on test-set: 98.8%
0 5.865943
100 0.10817811
200 0.11918729
300 0.07712402
400 0.15701644
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:12
0 0.061188698
100 0.05551771
200 0.03923799
300 0.061671436
400 0.09783531
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:12
Graph Saved
Accuracy on test-set: 98.8%
0 4.8679814
100 0.12691292
200 0.079303674
300 0.08100265
400 0.13900924
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:11
0 0.094664715
100 0.029472224
200 0.087135196
300 0.05876701
400 0.03957418
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:12
Graph Saved
Accuracy on test-set: 98.9%
0 5.082157
100 0.11163908
200 0.10599779
300 0.109804116
400 0.14786735
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:11
0 0.09514876
100 0.04610665
200 0.03292337
300 0.05942087
400 0.074609235
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:12
Graph Saved
Accuracy on test-set: 98.8%
Model Constructed in 1474.4931011199951 seconds
Accuracy on test-set: 99.0%
INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_0


Restoring parameters from graphs/cnn_mnist_0


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_1


Restoring parameters from graphs/cnn_mnist_1


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_2


Restoring parameters from graphs/cnn_mnist_2


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_3


Restoring parameters from graphs/cnn_mnist_3


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_4


Restoring parameters from graphs/cnn_mnist_4


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_5


Restoring parameters from graphs/cnn_mnist_5


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_6


Restoring parameters from graphs/cnn_mnist_6


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_7


Restoring parameters from graphs/cnn_mnist_7


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_8


Restoring parameters from graphs/cnn_mnist_8


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_9


Restoring parameters from graphs/cnn_mnist_9

INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_0


Restoring parameters from graphs/cnn_mnist_0


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_1


Restoring parameters from graphs/cnn_mnist_1


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_2


Restoring parameters from graphs/cnn_mnist_2


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_3


Restoring parameters from graphs/cnn_mnist_3


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_4


Restoring parameters from graphs/cnn_mnist_4


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_5


Restoring parameters from graphs/cnn_mnist_5


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_6


Restoring parameters from graphs/cnn_mnist_6


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_7


Restoring parameters from graphs/cnn_mnist_7


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_8


Restoring parameters from graphs/cnn_mnist_8


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_9


Restoring parameters from graphs/cnn_mnist_9



<Figure size 432x288 with 0 Axes>

INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_0


Restoring parameters from graphs/cnn_mnist_0


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_1


Restoring parameters from graphs/cnn_mnist_1


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_2


Restoring parameters from graphs/cnn_mnist_2


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_3


Restoring parameters from graphs/cnn_mnist_3


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_4


Restoring parameters from graphs/cnn_mnist_4


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_5


Restoring parameters from graphs/cnn_mnist_5


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_6


Restoring parameters from graphs/cnn_mnist_6


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_7


Restoring parameters from graphs/cnn_mnist_7


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_8


Restoring parameters from graphs/cnn_mnist_8


INFO:tensorflow:Restoring parameters from graphs/cnn_mnist_9


Restoring parameters from graphs/cnn_mnist_9



<Figure size 432x288 with 0 Axes>

[Back to top]
3 Adversarial CNN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
from data_loading import load_mnist
from model_CNN import CNN_adversary

%matplotlib inline


" Data Loading "
data = load_mnist()


" Model Construction "
tf.reset_default_graph()
g = tf.Graph() 
with g.as_default():
    with tf.Session(graph=g) as sess:
        with tf.variable_scope("cnn_mnist_adversary"):
            classifier = CNN_adversary(sess, data)
            classifier.graph_construction()
            tf.global_variables_initializer().run(session=sess)
            classifier.train()
            classifier.print_test_accuracy()

            # immunization
            for i in range(10):
                print("Generation of Adversarial Noise {0}".format(i))
                classifier.train_adversary(adversarial_target_cls=i)
                classifier.print_test_accuracy()
                classifier.plot_noise()
                classifier.print_confusion_matrix()

                print("Immunization to Adversarial Noise {0}".format(i))
                classifier.train()
                classifier.print_test_accuracy()

            # save
            classifier.save(sess=sess, save_path='graphs/cnn_mnist_adversary')
/Users/sungchul/anaconda/lib/python3.5/importlib/_bootstrap.py:222: RuntimeWarning: compiletime version 3.6 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.5
  return f(*args, **kwds)
/Users/sungchul/anaconda/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters


0 6.6573296
100 0.20184886
200 0.14317998
300 0.079005696
400 0.122229196
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:12
0 0.032132577
100 0.022789702
200 0.07079115
300 0.04498001
400 0.04390743
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:12
Accuracy on test-set: 98.9%
Generation of Adversarial Noise 0
0 13.6975355
100 1.1416681
200 0.9734287
300 0.7697133
400 0.69277143
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.125
Time Usage: 0:00:54
0 0.6333063
100 0.7526244
200 0.8084923
300 0.71366525
400 0.6513612
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.117188
Time Usage: 0:00:54
Accuracy on test-set: 16.3%

[[ 980    0    0    0    0    0    0    0    0    0]
 [1131    0    0    0    0    0    0    0    4    0]
 [ 849    0  183    0    0    0    0    0    0    0]
 [ 985    0    0   13    0    0    0    0   12    0]
 [ 938    0    0    0   31    0    0    0   12    1]
 [ 820    0    0    0    0   59    0    0   13    0]
 [ 869    0    0    0    0    0   89    0    0    0]
 [ 955    0    3    0    0    0    0   70    0    0]
 [ 796    0    0    0    0    0    0    0  178    0]
 [ 981    0    0    0    0    0    0    0    4   24]]

Immunization to Adversarial Noise 0
0 0.7948508
100 0.090543486
200 0.054149553
300 0.09189231
400 0.10754723
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:06
0 0.038108155
100 0.030287124
200 0.02123813
300 0.05231884
400 0.052929237
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:06
Accuracy on test-set: 98.8%
Generation of Adversarial Noise 1
0 15.463252
100 8.689096
200 7.914307
300 5.6727257
400 5.5194125
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.890625
Time Usage: 0:00:55
0 5.359372
100 6.1676664
200 6.826482
300 5.3386045
400 5.1174817
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.875
Time Usage: 0:00:55
Accuracy on test-set: 83.3%

[[ 789  153   33    0    1    0    0    2    1    1]
 [   0 1135    0    0    0    0    0    0    0    0]
 [   0   21 1010    0    0    0    0    1    0    0]
 [   0   76   15  911    0    0    0    3    2    3]
 [   0   73    3    0  906    0    0    0    0    0]
 [   0   66    1   15    0  792    0    4   11    3]
 [   2  306   13    0    5   13  587    0   32    0]
 [   0   81   67    0    1    0    0  878    0    1]
 [   0   46   30    0    2    2    0    2  892    0]
 [   0   88   15    0  384    1    0   68   25  428]]

Immunization to Adversarial Noise 1
0 0.12677337
100 0.029594751
200 0.0404158
300 0.05279514
400 0.11103601
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:07
0 0.037324768
100 0.021247696
200 0.051348336
300 0.02029874
400 0.05444051
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:07
Accuracy on test-set: 98.7%
Generation of Adversarial Noise 2
0 17.221222
100 11.388534
200 10.67425
300 7.1339087
400 6.691797
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.804688
Time Usage: 0:00:55
0 6.016178
100 5.1620045
200 6.384918
300 4.2357874
400 4.708252
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.734375
Time Usage: 0:00:54
Accuracy on test-set: 70.9%

[[ 765    0  214    0    0    0    0    0    1    0]
 [   0  388  747    0    0    0    0    0    0    0]
 [   0    0 1032    0    0    0    0    0    0    0]
 [   0    0  132  877    0    0    0    0    1    0]
 [   0    0  264    0  712    0    0    1    0    5]
 [   0    0   70   96    0  708    0    2   15    1]
 [   3    0  188    0    0    1  766    0    0    0]
 [   0    0  284    0    0    0    0  744    0    0]
 [   0    0  471    0    0    0    0    0  503    0]
 [   0    0  381   12    1    1    0    9    6  599]]

Immunization to Adversarial Noise 2
0 0.14942408
100 0.0740392
200 0.044807393
300 0.056573957
400 0.03106604
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:06
0 0.016945397
100 0.020439388
200 0.004117022
300 0.055193767
400 0.09208535
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.984375
Time Usage: 0:01:07
Accuracy on test-set: 98.8%
Generation of Adversarial Noise 3
0 19.974524
100 14.614485
200 12.768046
300 9.696026
400 8.98703
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.867188
Time Usage: 0:00:55
0 8.083714
100 7.6883807
200 7.7893863
300 6.4279456
400 6.280472
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.726562
Time Usage: 0:00:55
Accuracy on test-set: 73.9%

[[ 717    0    0  139    0   90    0    2    4   28]
 [   0 1026    0  105    0    1    1    0    1    1]
 [   0    1  431  597    0    2    0    0    1    0]
 [   0    0    0 1009    0    1    0    0    0    0]
 [   0    1    1  167  665    6    3    0   14  125]
 [   0    0    0   54    0  838    0    0    0    0]
 [   0    2    0   43    2  274  636    0    1    0]
 [   0    0    0  428    0    2    0  567    0   31]
 [   0    0    0  344    0   17    0    0  612    1]
 [   0    0    0  115    0    3    0    0    0  891]]

Immunization to Adversarial Noise 3
0 0.039295465
100 0.031307146
200 0.05612449
300 0.029746125
400 0.08710274
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:06
0 0.015863474
100 0.024241537
200 0.051463816
300 0.02818027
400 0.03208948
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:06
Accuracy on test-set: 98.8%
Generation of Adversarial Noise 4
0 22.869596
100 20.940884
200 17.557825
300 14.077925
400 10.100932
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.84375
Time Usage: 0:00:55
0 10.31779
100 10.204619
200 10.310134
300 8.210742
400 5.8814507
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.71875
Time Usage: 0:00:56
Accuracy on test-set: 72.9%

[[840   1   0   0 132   0   3   2   1   1]
 [  0 339   0   0 796   0   0   0   0   0]
 [  0   2 940   0  63   0   0   2  25   0]
 [  0   0  11 806 137  22   0  10  11  13]
 [  0   0   0   0 982   0   0   0   0   0]
 [  0   0   0   3 121 758   1   1   6   2]
 [  0   1   0   0 144   0 812   0   1   0]
 [  0   2   5   0 169   0   0 850   1   1]
 [  0   0   0   0 248   2   0   0 724   0]
 [  0   0   0   0 769   0   0   1   0 239]]

Immunization to Adversarial Noise 4
0 0.24244706
100 0.027763635
200 0.039670233
300 0.015568084
400 0.047790233
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:07
0 0.009460995
100 0.008744502
200 0.02915264
300 0.02748999
400 0.019252429
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:07
Accuracy on test-set: 98.8%
Generation of Adversarial Noise 5
0 21.827358
100 16.77768
200 14.728376
300 13.329803
400 12.55994
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.953125
Time Usage: 0:00:55
0 11.893584
100 9.867439
200 10.365736
300 10.807883
400 11.173431
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.945312
Time Usage: 0:00:54
Accuracy on test-set: 90.9%

[[ 936    0    0    0    0   43    0    0    1    0]
 [   0 1095    0    3    0   20    2    0   14    1]
 [   6    0  916   25    0   36    7    3   39    0]
 [   0    0    0  864    0  141    0    1    1    3]
 [   0    0    0    0  896   36   14    0    1   35]
 [   0    0    0    1    0  891    0    0    0    0]
 [   2    1    0    0    3   66  885    0    1    0]
 [   0    1    5   17    0   40    0  926   11   28]
 [   1    0    0    1    0  188    2    0  778    4]
 [   0    0    0    1    4  100    0    0    3  901]]

Immunization to Adversarial Noise 5
0 0.09530078
100 0.04566767
200 0.015136253
300 0.033816945
400 0.013883615
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:12
0 0.0011723869
100 0.061961755
200 0.010426571
300 0.0056899623
400 0.030762658
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:13
Accuracy on test-set: 98.9%
Generation of Adversarial Noise 6
0 25.552748
100 22.1038
200 22.711973
300 19.978996
400 19.537514
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.984375
Time Usage: 0:00:54
0 20.662924
100 17.433414
200 19.048634
300 16.100441
400 15.903351
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.976562
Time Usage: 0:00:54
Accuracy on test-set: 96.5%

[[ 944    1    0    0    0    0   33    0    1    1]
 [   0 1099    1    1    0    4   29    0    1    0]
 [   3    1 1006    0    1    1   12    3    5    0]
 [   1    1    2  940    0   62    1    1    2    0]
 [   0    0    0    0  951    1   23    0    2    5]
 [   0    0    0    1    0  874   13    0    4    0]
 [   0    0    0    0    0    0  958    0    0    0]
 [   1    2   13    4    1    2    2  990    1   12]
 [   3    0    2    1    0    4   31    2  930    1]
 [   1    1    1    0    6   30    6    0    3  961]]

Immunization to Adversarial Noise 6
0 0.064291775
100 0.031406194
200 0.0077121793
300 0.02592656
400 0.035185963
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:12
0 0.0023717314
100 0.010035193
200 0.0096188765
300 0.012409035
400 0.050495315
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:13
Accuracy on test-set: 98.9%
Generation of Adversarial Noise 7
0 25.20941
100 23.876144
200 23.092724
300 18.266716
400 19.708506
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.960938
Time Usage: 0:00:54
0 17.338943
100 17.893309
200 16.629234
300 14.554527
400 14.939243
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.875
Time Usage: 0:00:55
Accuracy on test-set: 91.2%

[[ 972    0    2    0    0    0    0    6    0    0]
 [   2  935    3    1    0    0    0  192    2    0]
 [   2    0  973    3    0    0    0   54    0    0]
 [   0    0    1  992    0    0    0   17    0    0]
 [   0    0    2    0  790    0    0  144    5   41]
 [   1    0    1   54    0  828    0    7    0    1]
 [  17    2    4    3    3   25  893    6    5    0]
 [   0    0    1    0    0    0    0 1027    0    0]
 [   8    0    9   46    0    1    0   29  878    3]
 [   1    0    0    5    1    2    0  161    2  837]]

Immunization to Adversarial Noise 7
0 0.18337722
100 0.01552662
200 0.04421819
300 0.022798928
400 0.05113214
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:07
0 0.009275484
100 0.016135762
200 0.019313231
300 0.012070845
400 0.0038853877
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:07
Accuracy on test-set: 98.9%
Generation of Adversarial Noise 8
0 22.60503
100 19.847326
200 21.633327
300 18.618467
400 17.574024
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:00:55
0 18.910147
100 17.70163
200 19.745443
300 17.015408
400 16.399847
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.984375
Time Usage: 0:00:55
Accuracy on test-set: 97.8%

[[ 973    0    2    0    0    0    1    0    3    1]
 [   1 1113    5    0    0    0    2    0   14    0]
 [   0    0 1026    0    1    0    0    1    4    0]
 [   0    0    9  975    0    4    0    2   18    2]
 [   0    0    4    0  954    0    3    0    6   15]
 [   0    0    0    4    0  871    1    0   14    2]
 [   7    2    2    0    1    1  940    0    5    0]
 [   0    5   36    0    0    0    0  971    7    9]
 [   1    0    1    0    0    0    0    1  970    1]
 [   2    0    0    0    2    1    0    2   16  986]]

Immunization to Adversarial Noise 8
0 0.018390473
100 0.0043595913
200 0.004126462
300 0.018652193
400 0.0101069575
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:01:06
0 0.044111595
100 0.03921738
200 0.0021831775
300 0.009406251
400 0.011551937
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:07
Accuracy on test-set: 99.0%
Generation of Adversarial Noise 9
0 28.360914
100 26.638948
200 25.396624
300 26.264736
400 21.779419
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:00:55
0 21.803083
100 21.864918
200 21.867998
300 23.560535
400 19.835958
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 0.992188
Time Usage: 0:00:55
Accuracy on test-set: 98.5%

[[ 974    0    0    0    1    0    1    1    1    2]
 [   0 1131    0    0    0    0    2    1    0    1]
 [   1    1 1013    0    4    0    2    7    3    1]
 [   0    0    2  989    0    5    0    3    4    7]
 [   0    0    0    0  967    0    0    0    0   15]
 [   0    1    0    2    0  887    0    1    1    0]
 [   2    1    0    0    8   13  933    0    1    0]
 [   0    2    5    0    1    0    0 1005    1   14]
 [   3    0    0    1    6    4    0    2  950    8]
 [   0    0    0    0    3    1    0    0    0 1005]]

Immunization to Adversarial Noise 9
0 0.007258728
100 0.013223343
200 0.016063038
300 0.005140857
400 0.027230298
==========================================================
Epoch: 0
Accuracy on Random Test Samples: 1
Time Usage: 0:01:07
0 0.0002702887
100 0.004416906
200 0.0016312511
300 0.017944986
400 0.0069409963
==========================================================
Epoch: 1
Accuracy on Random Test Samples: 1
Time Usage: 0:01:07
Accuracy on test-set: 99.0%
Graph Saved
