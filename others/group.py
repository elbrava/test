import json
import math

max_limit = 10000
vals = []


def primer(min, val_limit):
    vals = [v for v in range(min, val_limit)]
    for num in range(1, val_limit):
        print(num)
        l = []
        prime = True
        for x in range(2, num):
            l.append(x)

            if num % x == 0:
                print(num, "divsible", x)
                try:
                    vals.remove(num)
                except:
                    pass

    return vals


default_temp = {
    "members": [],
    "info": {
        "num_members": 0,
        "full": False
    }}

##print(primer(3, 1000))
# print()
import matplotlib as plt


def calc(num_dict, val, div):
    groups = val // div
    excess = [val % div, div - val % div]
    # print(excess)
    excess = min(excess)
    # print(excess)
    percent_loss = excess / div
    # print(div, groups, percent_loss)
    # print(percent_loss / groups)
    try:
        num_t = num_dict[percent_loss]
    except:
        num_dict[percent_loss] = []
        num_dict[percent_loss].append(div)
    else:
        num_dict[percent_loss].append(div)
        # print(percent_loss, excess, div, )
    return (groups, div, excess, percent_loss)


def group(num, exp):
    val = num * (100 + exp) / 100
    num_dict = {}
    val = math.ceil(val)
    num_list = []
    for div in [x for x in range(3, 9)]:
        _ = calc(num_dict, val, div)
    num_min = min([i for i in num_dict.keys()])
    error_max = max(num_dict[num_min])
    print(num_list)
    print(num_dict)
    print(error_max)
    return [calc(num_dict, val, error_max)]


def create_group():
    groups_dict = {}
    for groups, div, excess, loss in list(group(11000, 10)):
        for i in range(groups + 1):
            a = {
                "members": [],
                "info": {
                    "num_members": 0,
                    "full": False
                }}

            try:
                num_t = groups_dict["groups"]
            except:
                groups_dict["groups"] = []
                groups_dict["groups"].append({i: a})
            else:
                groups_dict["groups"].append({i: a})
                # print(percent_loss, excess, div, )

            groups_dict["info"] = {}
            groups_dict["info"]["num"] = groups
            groups_dict["info"]["divisions"] = div
            groups_dict["info"]["excess"] = excess
            groups_dict["info"]["percent_loss"] = loss
    with open("groups.json", "w") as w:
        json.dump(groups_dict, w)


# create_group()


# with open("groups.json", "w") as w:

def add_users_to_groups(users):
    with open("groups.json", "r") as f:
        f = json.load(f)
    all_not_full = True
    not_full = []
    empty = []
    div_dict = {}

    def check_full():

        for i, di in enumerate(f["groups"]):
            d = dict(f["groups"][i])[f"{i}"]
            if not d["info"]["full"]:
                not_full.append(di)
                print(di)
            if len(d["members"]) == 0:
                empty.append(di)
        for val in not_full:
            v = list(dict(val).keys())[0]
            v["info"]["num_members"] = len(v["members"])
            if div_dict[len(v["members"])]:
                div_dict[len(v["members"])].append(val)

            else:
                div_dict[len(v["members"])] = []
                div_dict[len(v["members"])].append(val)

    def form_group(users):
        div = f["info"]["divisions"]
        for i in range((len(users) // f["info"]["divisions"]) + 1):
            if not empty:
                d = dict(empty[0])
                d["members"].extend(users[(i - 1) * div:i * div])
                d["info"]["full"] = True
                empty.remove(empty[0])
                not_full.remove(empty[0])


            else:
                num = f["info"]["num"]
                f["groups"].append({num + 1: default_temp})
                f["info"]["num"] += 1
                empty.append(f["groups"][-1][f"{i}"])
                d = dict(empty[0])
                d["members"].extend(users[(i - 1) * div:i * div])
                d["info"]["full"] = True
                empty.remove(empty[0])

    def add_to_group(users):
        # needs testing

        div = f["info"]["divisions"]
        unable_to_add = False
        for i in range(1, len(users) + 1):
            print(i)
            if div_dict[i]:
                for user in div_dict[i]:
                    d = dict(user)[list(dict(user).keys())[0]]
                    d["members"].extend(users[0:i - 1])
                    for u in users[0:i - 1]:
                        users.remove(u)
                        div_dict[i].remove(u)

                    d["info"]["full"] = True
                    not_full.remove(user)

            else:
                unable_to_add = True

    check_full()
    if len(users) >> f["info"]["divisions"]:
        nums = len(users) // f["info"]["divisions"]
        nums_rem = len(users) % f["info"]["divisions"]

        form_group(users[0:nums])
    else:
        add_to_group(users[users])
        # check from five to join check for empty
# add user and further calculate
# additional groups
# based on entry
# exceeding info divisions
# below if % above 50%
# allow to form group
# one one one
# if all groups lack people
# users.lock(true)
# users told to wait
