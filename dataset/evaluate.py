import sys
import numpy as np

true_file, pred_file = sys.argv[1], sys.argv[2]

with open(f'{true_file}', 'r') as input_file:
    y_true = input_file.read().split(',')

with open(f'{pred_file}', 'r') as input_file:
    y_pred = input_file.read().split(',')

y_true = [float(y.strip()) for y in y_true]
y_pred = [float(y.strip()) for y in y_pred]

truths = []
preds = []
for i, true_value in enumerate(y_true):
    if not np.isnan(true_value):
        truths.append(true_value)
        preds.append(y_pred[i])

print(np.sqrt(
    np.square(
        np.subtract(truths, preds)).mean()))
