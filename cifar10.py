import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras. utils import to_categorical

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



for i in range(9):
    plt.subplot(330+1+i)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.imshow(x_train[i])
    
    
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32')/ 255.0
x_test = x_test.astype('float32')/255.0

print('y_train after categoricdal', y_train.shape)
print('y_test after categoricdal', y_test.shape)

plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')
plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')
	

'''# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
'''