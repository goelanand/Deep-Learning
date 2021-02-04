import tensorflow as tf
import keras
from tensorflow.keras.datasets import mnist
from keras import utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Preprocessing
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test = x_test.reshape(-1, 28,28,1).astype('float32')/255.0
#y_train = utils.to_categorical(y_train)
#y_test = utils.to_categorical(y_test)

print('done')

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        brightness_range=(0.1,0.5),
        shear_range = 0.2,
        channel_shift_range=0.3,
        fill_mode='nearest',
        cval=0.2,
        rescale=None,
        preprocessing_function=None,
        data_format=None, 
        validation_split=0.2,
        dtype=None)


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',
                 input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

print('done')

# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )

# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

lrs = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

rlp = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

%%time
batch_size = 64
epochs = 25

# Fit the Model
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), epochs = epochs, 
                              validation_data = (X_validation,y_validation), verbose=1, 
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks = [lrs])



result = pd.DataFrame(model.history.history)
# Plotting train 'loss' vs 'val_loss'
result[['loss','val_loss']].plot(figsize=(8, 5))
plt.grid(True)
plt.show()

# Plotting train 'accuracy' vs 'val_accuracy'
result[['accuracy','val_accuracy']].plot(figsize=(8, 5))
plt.grid(True)
plt.show()

print(model.evaluate(X_validation, y_validation))


from sklearn.metrics import classification_report,confusion_matrix

# Instead of probabilities it provides class labels
y_pred_classes = model.predict_classes(X_validation)

# Reverting one-hot encoding on true validation output labels
y_test_classes = np.argmax(y_validation,axis=1)
print("################ CLASSIFICATION REPORT ################")
print(classification_report(y_test_classes,y_pred_classes),"\n\n")
print("################ CONFUSION MATRIX ################")
plt.figure(figsize=(8,8))
sns.heatmap(confusion_matrix(y_test_classes,y_pred_classes),linewidths=.5,cmap="YlGnBu",annot=True,cbar=False,fmt='d')
plt.show()


# Performing prediction on unseen data
pred_digits_test = np.argmax(model.predict(X_test),axis=1)
# alternate method : pred_digits_test = model.predict_classes(X_test)

# Saving result to .csv file for final submission
image_id_test=[]
for i in range (len(pred_digits_test)):
    image_id_test.append(i+1)
d={'ImageId':image_id_test,'Label':pred_digits_test}
answer=pd.DataFrame(d)



