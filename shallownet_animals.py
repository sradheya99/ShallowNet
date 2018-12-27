from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ShallowCNNModule.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from ShallowCNNModule.preprocessing.simplepreprocessor import SimplePreprocessor
from ShallowCNNModule.datasets.simpledatasetloader import SimpleDatasetLoader
from ShallowCNNModule.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, help = 'path to dataset')
arg = vars(ap.parse_args())

print('[INFO] Loading Images....')
imagePaths = list(paths.list_images(arg['dataset']))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors = [sp, iap])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype('float') / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

trainY = LabelEncoder().fit_transform(trainY)
testY = LabelEncoder().fit_transform(testY)

trainY = to_categorical(trainY)
testY = to_categorical(testY)

print('[INFO] Compiling Model....')
opt = SGD(lr = 0.005)
model = ShallowNet.build(width = 32, height = 32, depth = 3, classes = 2)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

print('[INFO] Training the Network....')
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 32, epochs = 100, verbose = 1)

print('[INFO] Evaluating Network....')
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = ['cat', 'dog']))

plt.style.use('ggplot')
plt.figure(figsize = (8, 5))
plt.plot(np.arange(0, 100), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label = 'val_acc')

plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()