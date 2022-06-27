import datetime
import threading

import pandas as pd

from telegram.Interest import interest_dict

expenditure = {
    "name": "salary",
    "value": 700,
    "in": True,
    "must": True,
    "payment_func": "add",
    "created": "date",
    "due": "duration",

}
period_expenditure = {
    "name": "february",
    "expediets": ["list of expenditures"]
}
overheads = {
    "name": "2007",
    "periodic_expe": ["list of totals"]
}


class Expenditure(object):
    def __init__(self, value, name, payment_func="monthly", interest_func="simple", interest=0, loan=False, dur=40):
        super(Expenditure, self).__init__()
        self.loan = loan
        self.name = name
        self.value = value
        self.periodic_value = self.value
        self.payment_func = payment_func
        self.interest_func = interest_func
        # once,monthly,yearly,weekly,daily

        self.interest = interest

        self.created = datetime.date.today()
        self.duration = dur
        # days

    def __str__(self):
        loaning = "credit"
        if self.loan:
            loaning = "debit"
            return f"{self.name} {loaning}  of {self.value} at {self.interest} {self.payment_func} from {self.created} for {self.duration} days"
        else:
            return f"{self.name} {loaning} of {self.value} {self.payment_func}"

    def val(self, date):
        duration = self.created - date
        duration = duration.days
        t = threading.Thread(target=interest_dict[self.interest_func], args=[self, duration, self.payment_func])
        t.start()
        t.join()
        return self.periodic_value

    def auto_check(self):
        if self.loan:
            self.value = self.val(datetime.date.today())
            if (datetime.date.today() - self.created).days >= self.duration: del self


class Periodic_Exp(object):
    def __init__(self, name, exps):
        super(Periodic_Exp, self).__init__()
        self.name = name
        self.exps = exps
        self.value_series = pd.Series(self.exps, index=[e.name for e in exps])
        self.total = sum(exps)
        if self.total > 0:
            self.balanced = False
        else:
            self.balanced = True

    def plot(self):
        self.value_series.plot()

    def update(self):
        for e in self.exps:
            e.auto_check()
        self.value_series = pd.Series(self.exps, index=[e.name for e in self.exps])
        self.total = sum(self.exps)
        if self.total < 0:
            self.balanced = False
        else:
            self.balanced = True

        # data frame
        # plot
        # describe sum
        # below above

        # easy modelling of sys-


class Overhead(object):
    def __init__(self, name, periods):
        super(Overhead, self).__init__()
        self.name = name
        self.periods = periods
        self.value = 0
        self.value_series = pd.concat(objs=[e.exps for e in periods])
        totals = []
        index = []
        for e in periods:
            totals.append(e.total)
            index.append(e.name)
        self.total = sum(totals)
        self.expenditure_series = pd.Series(totals, index=index)
        if self.total < 0:
            self.balanced = False
        else:
            self.balanced = True
        threading.Thread(target=self.update).start()

    def update(self):
        for e in self.periods:
            e.update()
        self.value_series = pd.concat(objs=[e.exps for e in self.periods])
        totals = []
        index = []
        for e in self.periods:
            totals.append(e.total)
            index.append(e.name)
        self.total = sum(totals)
        self.expenditure_series = pd.Series(totals, index=index)


