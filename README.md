# Scripts
These are the scripts used in all my experiments. 
Some may contain code which imports different datasets or handle different sizes of data. 
It allowed me to be flexible, e.g. quick testing.

Names of the scripts reflect the model or their architecture, e.g. VGG-GBC.py loads the profiles & uses the VGG representation which is then passed to the GBC.

# Changes
#### Bins
```python
# 3 bins
quantiles = [0, 0.15, 0.85, 1]


# 20 bins
quantiles = [x/100 for x in range(0, 101, 5)]
```

#### Monitored quanties for earlystop and checkpoint

```python
# 3 bins - accuracy
early_stop = EarlyStopping(monitor='val_top_k', min_delta=0, patience=20, verbose=1)
fp_checkpoint = 'weights/best_weights.h5'
checkpoint = ModelCheckpoint(fp_checkpoint, monitor='val_loss', verbose=1, save_best_only=True)


# 20 bins - mae
early_stop = EarlyStopping(monitor='val_mean_absolute_error', min_delta=0, patience=10, verbose=1)
fp_checkpoint = 'weights/best_weights.h5'
checkpoint = ModelCheckpoint(fp_checkpoint, monitor='val_mean_absolute_error', verbose=1, save_best_only=True)
```

#### Saving metrics to file instead of printing to console
```python
fs = open('metrics.txt', 'w')
fs.write('Acc val: ' + str(metrics.accuracy_score(y_val, y_pred_val)) + '\n')
fs.write('MAE val:  ' + str(metrics.mean_absolute_error(y_val, y_pred_val)) + '\n')
fs.write('CF val:  ' + str(metrics.confusion_matrix(y_val, y_pred_val)) + '\n')
fs.write('Acc test: ' + str(metrics.accuracy_score(y_test, y_pred_test)) + '\n')
fs.write('MAE test:  ' + str(metrics.mean_absolute_error(y_test, y_pred_test)) + '\n')
fs.write('CF test:  ' + str(metrics.confusion_matrix(y_test, y_pred_test)) + '\n')
fs.close()
```
