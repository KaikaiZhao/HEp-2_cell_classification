""" Fine tune VGG16 model on cell/dataset2 and extract features. The dataset is split into
(train, validation, test)=(64%, 16%, 20%). The best accuracy is about 96.7%. The accuracy of SVM trained on features
extracted from 'fc1' layer of fine tuned VGG model can come to 97.35%.

Data format:
    images: numpy.ndarray, like images.npy, shape=(nb_samples, widht, height, 3)
    labels: numpy.ndarray, like lables.npy, shape=(nb_samples, )
Examples:
    1. Extract VGG16 features
        python vgg16_classify_cell.py --mode extract_vgg_feature --data_dir YOUR_DATA_DIR
    2. Extract features from fine tuned model. Suppose the saved model is named of 'my_weights.h5'
        python vgg16_classify_cell.py --mode extract_feature --trained_model_path my_weights.h5
    3. Fine tune the model, there should be sub folders named 'checkpoint' and 'logs' in current dir.
        python vgg16_classify_cell.py --mode train
    4. Test the fine tuned model on testing data set
        python vgg16_classify_cell.py --mode test --trained_model_path my_weights.h5
"""
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
import datetime
import scipy.io as sio
from keras.utils import plot_model
import h5py 

now = datetime.datetime.now

# parameters can be passed from user's console
tf.app.flags.DEFINE_string('data_dir', '~/medical/exp2_finetune_vgg2',
                           'the dataset dir')
tf.app.flags.DEFINE_integer('input_size', 224, 'the width/height of input image')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'one of {train, test, extract_feature, extract_vgg_feature}')
tf.app.flags.DEFINE_string('trained_model_path', None,
                           'If the mode!=train, the directory of trained model should be given')
FLAGS = tf.app.flags.FLAGS


def my_model(finetune=False, nb_class=None):
    """ Define the model

    Arguments:
        finetune: False, the default VGG16 model to extract feature;
                  True, add some dense layers to classifier new data set
        nb_class: number of classes, only valid when finetune=True
    Return:
        model: defined model
    """
    # This will download weights of vgg16 trained on ImageNet
    #            'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # to  ~/.keras/model/
    vgg16_model = VGG16(input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                        weights='imagenet', include_top=False)

    # Add new layer after VGG16 base model
    added = vgg16_model.output
    added = Flatten(name='flattened')(added)  # can extract feature from here

    if finetune:
        added = Dense(1024, activation='relu', name='fc1')(added)  # features here are better after fine tuned
        added = Dense(1024, activation='relu', name='fc2')(added)  # features here are better after fine tuned
        #added = Dropout(0.5)(added)  # if tend to over fitting, use dropout
        added = Dense(nb_class, activation='sigmoid')(added)

    model = Model(input=vgg16_model.input, output=added)
    plot_model(model,to_file='model.png', show_shapes=True)
    """
    # fix the weights of VGG16 model
    for layer in vgg16_model.layers:
        layer.trainable = False
    """
    model.compile(Adam(lr=0.00001), 'binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, save_pre=None):
    """ Given the predefined model, use the training data to train the model

    Arguments:
        model: type, keras.models.Model; the predefined model to be trained.
        save_pre: the prefix file name for saving the training records
    Return:
        model: the trained model
    """
    # load data
    picnum = 721
    print('loading training data')
    #import h5py
    #img = sio.loadmat(FLAGS.data_dir + '/tranx9.mat')  # shape=(13596, 100, 100, 3), pixel in [0, 256)
    #img = sio.loadmat('tranx9.mat')  # shape=(13596, 100, 100, 3), pixel in [0, 256)
    #label = sio.loadmat(FLAGS.data_dir + '/label2.mat')  # shape=(13596,), value in {1,2,3,4,5,6}
    label = sio.loadmat('y_label.mat')  # shape=(13596,), value in {1,2,3,4,5,6}
    
    print('aaa')
 #   dict_data = h5py.File('tranx9.mat');
    img0 = np.load('images_dataset2_train4.npy');
    #img0 = img['tranx9']
#    img0 = dict_data['tranx9']
    print(img0.shape)
#    img0=np.transpose(img0,[3,2,1,0])
#    print(img0.shape)
    labels = label['y_label']
    #labels = np.reshape(labels,1,230)
    labels =  (labels)  # one-hot code, shape=(13596, 6), value in {0, 1}
    labels = to_categorical(labels)  # one-hot code, shape=(13596, 6), value in {0, 1}
    dict_data = np.reshape(img0, (picnum, 224, 224,3))
    # input()
    """
    imgs = np.zeros((picnum, 224, 224, 3))
    imgs[:, :, :, 0] = img0
    imgs[:, :, :, 1] = img0
    imgs[:, :, :, 2] = img0
    """
    imgs = img0

    print (imgs.shape)
    print (labels.shape)
    #  input()

    # if no index available, use the following two lines to generate and save new indices
    # index = np.random.permutation(len(labels))
    # np.save('index_shuffle_dataset2.npy', index)
    #imgs = imgs[:200]  # shuffled training images
    #labels = labels[:200]  # corresponding shuffled training labels

    nb_train_samples = 2700  # number of samples used to train the model, rest will serve as validation samples
    nb_samples =picnum
    # Augment data
    train_datagen = ImageDataGenerator( horizontal_flip=True, vertical_flip=True,
                                       shear_range=0.2, zoom_range=0.2, rotation_range=30.,
                                       width_shift_range=0.2, height_shift_range=0.2)
    vali_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(imgs[:nb_train_samples], y=labels[:nb_train_samples],
                                         batch_size=FLAGS.batch_size, shuffle=True)
    #input()
    vali_generator = vali_datagen.flow(imgs[nb_train_samples:], labels[nb_train_samples:],
                                       batch_size=FLAGS.batch_size)
    print('data prepared!')
    print('training samples: %d\nvalidation samples: %d'%(nb_train_samples, nb_samples - nb_train_samples))

    """
    # Augment data from directory
    train_generator = train_datagen.flow_from_directory(
        os.path.join(FLAGS.data_dir, 'train'),
        target_size=(FLAGS.input_size, FLAGS.input_size),
        color_mode='rgb',
        batch_size=FLAGS.batch_size)
    vali_generator = vali_datagen.flow_from_directory(
        os.path.join(FLAGS.data_dir, 'test'),
        target_size=(FLAGS.input_size, FLAGS.input_size),
        color_mode='rgb',
        batch_size=FLAGS.batch_size)
    """

    # Some functions used to record the behavior of training process. If feels boring, free to delete them

    # begin training
    print('start training...')
    t = now()
    history = model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=300,
                                  validation_data=vali_generator, nb_val_samples=nb_samples-nb_train_samples)
    print("Training finished!")
    print('Training time: %s' % (now() - t))
    print('Training history:')
    print(history.history)

    return model


def extract_features(model, images, model_path=None, name='fc1'):
    """ extract features

    Arguments:
        model: defined model
        model_path: the path of saved weights of model. 'None' for VGG default imagenet weights
        images: ndarray with shape=(,100,100,3) or the path of images
        name: the name of layer from which extract features
    Return:
        features: extracted features
    """
    picnum = 3431
    if type(images) is str:  # path of images
        #import h5py
        img = sio.loadmat(FLAGS.data_dir + '/tranx9.mat')  # shape=(13596, 100, 100, 3), pixel in [0, 256)
        label = sio.loadmat(FLAGS.data_dir + '/label2.mat')  # shape=(13596,), value in {1,2,3,4,5,6}
        # dict_data = h5py.File('tranx9.mat')
        img0 = img['tranx9']
        labels = label['lab2']
        # labels = np.reshape(labels,1,230)
        labels = (labels)  # one-hot code, shape=(13596, 6), value in {0, 1}
        labels = to_categorical(labels)  # one-hot code, shape=(13596, 6), value in {0, 1}
        dict_data = np.reshape(img0, (picnum, 224, 224))
        images = np.zeros((picnum, 224, 224, 3))
        images[:, :, :, 0] = img0
        images[:, :, :, 1] = img0
        images[:, :, :, 2] = img0
        #images = img0

    print (images.shape)
    #input()
    if model_path is not None:
        model.load_weights(model_path)
    feature_model = Model(model.input, model.get_layer(name=name).output)
    features = feature_model.predict(images)
    return features


def test_model(model, model_path):
    """ Use the trained model to test new samples

    Arguments:
        model: type, keras.models.Model, predefined model
        model_path: the full path of saved weights of trained model
    Return:
        result: (score, accuracy)
    """
    print('loading weights...')
    model.load_weights(model_path)

    print('loading testing data...')
    images = np.load(FLAGS.data_dir + '/images.npy')  # shape=(,100,100,3), rescale to [0,1]
    labels = np.load(FLAGS.data_dir + '/labels.npy')  # shape=(13596,), value in {1,2,3,4,5,6}
    index = np.load('index_shuffle_dataset2.npy')  # the seed to shuffle data, np.permutation(len(labels))
    images = images[index[10877:]]  # shuffled testing images
    labels = labels[index[10877:]]  # corresponding shuffled testing labels
    labels = to_categorical(labels)  # one-hot code, shape=(13596, 6), value in {0, 1}
    print (labels)
    input()
    print('begin testing')
    result = model.evaluate(images, labels, batch_size=FLAGS.batch_size)

    return result


def main(_):
    save_pre = 'vgg_cell_finetune1234'
    model = my_model(True,7)
    if FLAGS.mode == 'train':
        model = train_model(model, save_pre)
        model.save(save_pre + '_model.h5')
        print('model saved!')

    elif FLAGS.mode == 'test':
        print(testing)
        if FLAGS.trained_model_path is None:
            result = test_model(model, save_pre+'_model.h5')
        else:
            result = test_model(model, FLAGS.trained_model_path)
        print('Test score:', result[0])
        print('Test accuracy', result[1])

    elif FLAGS.mode == 'extract_feature':
        features = extract_features(model, model_path=FLAGS.trained_model_path, images=FLAGS.data_dir + '/tranx9.mat')
        print (features.shape)
        input()
        print('saving features...')
        sio.savemat('features_' + save_pre,{'feature':features})
        print('features saved to: features_' + save_pre)

    elif FLAGS.mode == 'extract_vgg_feature':
        features = extract_features(my_model(), images=FLAGS.data_dir + '/images.npy', name='flattened')
        print('saving features...')
        np.save('features_vgg.npy', features)
        print('features saved to: features_vgg.npy')

if __name__ == '__main__':
    tf.app.run()


"""
#  predict for a single image
img_path = 'ak47.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print features"""
