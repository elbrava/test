uni_list = []


def new_add(val):
    uni_list.append(val)


def cum():
    cum_dict = {}
    for u in uni_list:
        for k in dict(u).keys():
            cum_dict[k] += u[k]
    for val in cum_dict.keys():
        cum_dict[val] /= uni_list
