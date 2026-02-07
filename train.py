import pandas as pd
import numpy as np
from Model import Model

# loading  data
train_df = pd.read_csv("train_df.csv")
test_df = pd.read_csv("test_df.csv")
sub_df = pd.read_csv("submission.csv")

X_train = train_df.drop(columns=["ID", "num_errors"]).values
y_train = train_df["num_errors"].values
X_test = test_df.drop(columns=["index"]).values

# train
model = Model(n_components=50,lr=0.05,epochs=600,reg=1e-3)

model.fit(X_train, y_train)

# predict
preds = model.predict(X_test)

# set submission
sub_df["Predicted"] = preds
sub_df.to_csv("submission.csv", index=False)

print("Submission file saved.")