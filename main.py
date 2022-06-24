# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from ludwig.api import LudwigModel
import pandas

df = pandas.read_csv('train.csv')
model = LudwigModel(config='rotten_tomatoes.yaml')
results = model.train(dataset=df)