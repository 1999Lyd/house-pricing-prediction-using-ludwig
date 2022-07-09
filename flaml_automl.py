
import pandas as pd
from flaml import AutoML

input_df = pd.read_csv("data/train.csv")
X_train = input_df.drop(["SalePrice"], axis=1)
y_train = input_df["SalePrice"]  # pylint: disable=E1136



automl = AutoML()

settings = {
    "time_budget": 120,  # total running time in seconds
    "metric": "r2",  # can be: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
    # 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
    "task": "regression",  # task type
    "log_file_name": "house_price.log",  # random seed
}

automl.fit(X_train=X_train, y_train=y_train, **settings)

# retrieve best config and best learner
print("Best ML leaner:", automl.best_estimator)
print("Best hyperparmeter config:", automl.best_config)
print("Best accuracy on validation data: {0:.4g}".format(1 - automl.best_loss))
print(
    "Training duration of best run: {0:.4g} s".format(
        automl.best_config_train_time
    )
)