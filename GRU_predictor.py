import numpy as np
from matplotlib import pyplot as plt
from keras import layers
from keras.optimizers import RMSprop
from math import sqrt
from keras.layers import LSTM
import numpy as np
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


f=open('Folds5x2_pp.csv')
data=f.read()
f.close()
lines=data.split('\n')
header=lines[0].split(',')
lines=lines[1:9569]

float_data=np.zeros(((len(lines)),len(header)))
for i, line in enumerate(lines):
	values=[float(x) for x in line.split(',')[0:]]
	float_data[i,:]=values

mean=float_data[:7177].mean(axis=0)
float_data-=mean
std=float_data[:7177].std(axis=0)
float_data/=std


def generator(data, lookback, delay, min_index, max_index,shuffle=False, step=1, batch_size=128):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

val_steps=(7656-5743-24)
test_steps=(len(float_data)-7657-24)

train_gen=generator(float_data,lookback=24,delay=1,min_index=0,max_index=5742,shuffle=False,step=1,batch_size=128)
val_gen=generator(float_data,lookback=24,delay=1,min_index=5743,max_index=7656,shuffle=False,step=1,batch_size=128)
test_gen=generator(float_data,lookback=24,delay=1,min_index=7657,max_index=9569,shuffle=False,step=1,batch_size=128)

print(float_data.shape[-1])


model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(64,return_sequences=True, dropout=0.6, recurrent_dropout=0.3))
model.add(layers.GRU(32, dropout=0.6, recurrent_dropout=0.3))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop',loss='mae',metrics=['accuracy'])
monitor=EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=500,verbose=0,mode='auto')
checkpointer=ModelCheckpoint(filepath="model.h5",verbose=0,save_best_only=True)

model.save_weights("model.h5")
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=3,validation_data=val_gen,validation_steps=val_steps)
model.load_weights('model.h5')
test_loss, test_acc = model.evaluate(test_X, test_y)


pred=network.predict(test_X)
pred=np.argmax(pred,axis=1)
y_test=np.argmax(test_y,axis=1)
cm=confusion_matrix(y_test,pred)
np.set_printoptions(precision=2)
print('Confusion Matrix')
print(cm)
plt.figure()

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()






















