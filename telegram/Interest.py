import datetime
import os
import pathlib
from functools import partial

import pandas as pd


def simple(object, duration, payment_func):
    object.periodic_value = 0
    print("partial")


# data from model
print("here")

interest_dict = {
    "simple": simple,
}


def model_create():
    di = {"start_time": {"payment_time": "month",
                         "start_time": datetime.date.today(),
                         "end_time": "",
                         "amount_before": 0,
                         "payment_func": "0",
                         "amount_after": 0,
                         }}
    di["start_time"]["amount_after"] = ""
    test_path = "works.csv"
    if os.path.exists(test_path):
        print("exists")
        p = pd.read_csv(test_path, index_col=[0])
        print(len(p.values))
        i = int(len(p.values))
        dis = pd.DataFrame(di, index=[i])
        p = pd.concat([p, dis])
        print(p.head())
        p.to_csv(test_path)
    else:
        p = pd.DataFrame(di, index=[0])
        p.to_csv(test_path)


def getfrompoint():
    pass
    # exists return value
    # else not found
