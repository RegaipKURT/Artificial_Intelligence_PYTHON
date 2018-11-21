#Predicting cryptocurrency prices and deciding which one is has more advantage.

import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

def classify(current,future):
	if float(future) > float(current):
		return 1
	else:
		return 0

def preprocess_df(df):
	df = df.drop("future", 1)

	for i in df.columns:
		if i != "target":
			df[i] = df[i].pct_change()
			df.dropna(inplace=True)
			df[i] = preprocessing.scale(df[i].values)

	df.dropna(inplace=True)

	sequential_data = []
	prev_days = deque(maxlen= SEQ_LEN)

	for i in df.values:
		prev_days.append([n for n in i[:-1]])
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days), i[-1]])

	random.shuffle(sequential_data)

	buys = []
	sells = []

	for seq, target in sequential_data:
		if target == 0:
			sells.append([seq, target])
		elif target == 1:
			buys.append([seq, target])
	
	random.shuffle(buys)
	random.shuffle(sells)

	lower = min(len(buys), len(sells))

	buys = buys[:lower]
	sells = sells[:lower]

	sequential_data = buys + sells
	random.shuffle(sequential_data)

	X = []
	y = []

	for seq, target in sequential_data:
		X.append(seq)
		y.append(target)

	return np.array(X), y

main_df = pd.DataFrame()

ratios = ["BTC-USD","LTC-USD","ETH-USD","BCH-USD"]

for ratio in ratios:	
	dataset = f"crypto_data/{ratio}.csv"
	
	df = pd.read_csv(dataset,names=["time","low","high","open","close","volume"])
	df.rename(columns={"close": f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace=True)
	print ("{0}:\n".format(ratio),df.head(),"\n")

	df.set_index("time",inplace=True)
	df = df[[f"{ratio}_close",f"{ratio}_volume"]]
	print(df.head())

	if len(main_df) == 0:
		main_df = df
	else:
		main_df = main_df.join(df)

main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"],main_df["future"]))

print(main_df[[f"{RATIO_TO_PREDICT}_close","future","target"]].head(15))

times = sorted(main_df.index.values)
last_5prc = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5prc)]
main_df = main_df[(main_df.index < last_5prc)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train_data: {len(train_x)}, validation: {len(validation_x)}")
print(f"Don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"Validation dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
