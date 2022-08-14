from flask import Flask,render_template,request
import yahoo_fin.stock_info as si
import yfinance as yf
from yahoo_fin import options
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import preprocessing


app=Flask(__name__)
@app.route("/",methods=['POST','GET'])


def func():
    colours=si.tickers_nifty50()



    return render_template("home.html", colours=colours)




@app.route("/predict", methods=['GET', 'POST'])
def predict():
    input_data = request.form['Stocks']
    output=str(input_data)

    df = si.get_data(output).reset_index()
    df1 = df.iloc[:, :7]
    df1.drop(['adjclose'], axis=1, inplace=True)
    df1.ffill(inplace=True)
    df1 = df1.reindex(index=df1.index[::-1])
    obs = np.arange(1, len(df1) + 1, 1)
    OHLC_avg = df1.mean(axis=1)
    HLC_avg = df1[['high', 'low', 'close']].mean(axis=1)
    close_val = df1[['close']]

    # PREPARATION OF TIME SERIES DATASE
    OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg), 1))  # 1664
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg = scaler.fit_transform(OHLC_avg)

    # TRAIN-TEST SPLIT
    train_OHLC = int(len(OHLC_avg) * 0.75)
    test_OHLC = len(OHLC_avg) - train_OHLC
    train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC, :], OHLC_avg[train_OHLC:len(OHLC_avg), :]

    # TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
    trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
    testX, testY = preprocessing.new_dataset(test_OHLC, 1)

    # RESHAPING TRAIN AND TEST DATA
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    step_size = 1

    # LSTM MODEL
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, step_size), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.add(Activation('linear'))

    # MODEL COMPILING AND TRAINING
    model.compile(loss='mean_squared_error', optimizer='adagrad')  # Try SGD, adam, adagrad and compare!!!
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

    # PREDICTION
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # DE-NORMALIZING FOR PLOTTING
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
    trainPredictPlot = np.empty_like(OHLC_avg)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[step_size:len(trainPredict) + step_size, :] = trainPredict

    # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
    testPredictPlot = np.empty_like(OHLC_avg)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (step_size * 2) + 1:len(OHLC_avg) - 1, :] = testPredict

    # PREDICT FUTURE VALUES
    last_val = testPredict[-1]
    last_val_scaled = last_val / last_val
    next_val = model.predict(np.reshape(last_val_scaled, (1, 1, 1)))
   # print("Last Day Value:", np.asscalar(last_val))
    print("Next Day Value:", np.asscalar(last_val * next_val))
    # print np.append(last_val, next_val)
    out=np.asscalar(last_val * next_val)
    Predicted='Predicted Price tommorrow:'



    return render_template('home.html', Predicted=Predicted,prediction_text= out)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30006, debug=True)
