from ludwig.api import LudwigModel
import pandas

model = LudwigModel.load('results/experiment_run/model')

predictions, _ = model.predict(dataset='data/test.csv')
print(predictions.head())
