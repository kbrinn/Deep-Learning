from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
dataset = read_csv('Folds5x2_pp.csv', header=0)
values = dataset.values
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)


values = reframed.values

train = values[:6000, :]
test = values[6000:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = Sequential()
model.add(LSTM(96, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(96))
model.add(Dense(1))
model.compile(optimizer='rmsprop',loss='mae',metrics=['accuracy'])
monitor=EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=500,verbose=0,mode='auto')
checkpointer=ModelCheckpoint(filepath="model.hdf5",verbose=0,save_best_only=True)
model.save_weights("model.h5")
history=model.fit(train_X, train_y, validation_data=(test_X,test_y),callbacks=[monitor,checkpointer],verbose=2,epochs=10, batch_size=12)
model.load_weights('model.hdf5')
test_loss, test_acc = model.evaluate(test_X, test_y)

pred=model.predict(test_X)
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
























