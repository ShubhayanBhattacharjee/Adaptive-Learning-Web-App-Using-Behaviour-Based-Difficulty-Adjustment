import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Load dataset
df = pd.read_csv("data/final_dataset.csv")

seq_len = 5
data = df.values

# Create sequences
seq = []
for i in range(len(data)-seq_len):
    seq.append(data[i:i+seq_len])

seq = np.array(seq)

X = seq[:,:,:3]
y = seq[:,-1,1]

# Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Save split
os.makedirs("data/splits", exist_ok=True)
np.save("data/splits/X_test_dl.npy", X_test)
np.save("data/splits/y_test_dl.npy", y_test)

# Build model
model = keras.Sequential([
    keras.layers.LSTM(64,input_shape=(X.shape[1],X.shape[2])),
    keras.layers.Dense(32,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=5,batch_size=32)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/dkt_model.h5")

print("DL model trained")