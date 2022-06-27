import json
import os
import pathlib
import threading

import requests
import telebot
import pandas as pd
import time
import sys

sys.setrecursionlimit(100000000)
time_begin = time.perf_counter()
api_key = "5211132462:AAGEQo5ueNYnnXVrxz7kN0wtoCEGvnr-1lM"
bot = telebot.TeleBot(api_key)
print(bot)
error_msg = """
Commands
    /all- all animals
    /work - sends gathered images
    animal - select animals
        animal|<selected-animal>
        without "< >"
"""

vals_list = []


@bot.message_handler(commands=["start"])
def start(message):
    di = {
        "id": message.chat.id,
        "animal": "animal",
        "counter": 0,
        "ratio": 0.51,
    }
    test_path = "tg.csv"
    if os.path.exists(test_path):
        print("exists")
        p = pd.read_csv(test_path)
        print(len(p.values))
        i = int(len(p.values))
        dis = pd.DataFrame(di, index=[0])
        p = pd.concat([p, dis])
        print(p.head())
        p.to_csv(test_path, index=False)
    else:
        os.makedirs(pathlib.Path(test_path).parent, exist_ok=True)
        p = pd.DataFrame(di, index=[0])
        p.to_csv(test_path, index=False)

    bot.reply_to(message, error_msg)


@bot.message_handler(commands=["all"])
def all_animals(message):
    bot.reply_to(
        message,
        """
    CATS
    Other animals will be introduced later
    
    """,
    )


def message_handle(message):
    if message.text.__contains__("<") or message.text.__contains__(">"):
        bot.reply_to(message, error_msg)
        bot.reply_to(message, "< or > were found in message")
        return False
    else:
        if message.text.__contains__("|"):
            return True
        else:
            return False


@bot.message_handler(func=message_handle)
def animal(message):
    m = message.text.split("|")[-1]
    animal = m
    p_org = pd.read_csv("tg.csv", index_col=0)
    print(animal)
    v = len(p_org.values)
    for i in range(v):
        print(i)
        p = p_org.iloc[i]
        print(p)
        print(p[0])
        print(p[1])
        print(p[2])
        if p[0] == message.chat.id:
            print(p[1])
            p[1] = animal
            p_org.iat[i, 1] = animal
            p_org.to_csv("tg.csv", index=False)

    # p.iat[i, 2] = animal


@bot.message_handler(commands="work")
def timer(message):
    global time_begin
    print(message.text)
    time_now = time.perf_counter()
    print(time_begin)
    print(time_now)
    if time_now - time_begin > 1:
        thread_send_img()
        time_begin = time.perf_counter()
    time.sleep(30)
    timer(message)


def thread_send_img():
    print("called")
    p = pd.read_csv("tg.csv", index_col=0)
    v = len(p.values)

    for i in range(v):
        threading.Thread(target=send_img, args=[i]).start()


def send_img(i):
    p_org = pd.read_csv("tg.csv")
    global vals_list
    print(p_org)
    v = len(p_org.values)
    url = f"https://cat-fact.herokuapp.com/facts/random?animal_type=cat&amount=1"
    response = requests.request("GET", url)
    print(response.text)
    j = json.loads(response.text)
    print(j)
    text = j["text"]

    p = p_org.iloc[i]
    print(j["status"]["verified"])

    try:
        bot.send_photo(
            chat_id=int(p[0]),
            photo=json.loads(
                requests.request(
                    "GET", "https://api.thecatapi.com/v1/images/search"
                ).text
            )[0]["url"],
        )
        bot.send_message(int(p[0]), "HAPPY BIRTHDAY LOVE YOU")
        bot.send_message(int(p[0]), "\n<i><b>" + text + "</b></i>", parse_mode="HTML")
        bot.send_message(
            int(p[0]),
            "\n<i><b> IS FACT : " + str(j["status"]["verified"]).upper() + "</b></i>",
            parse_mode="HTML",
        )

    except Exception as e:
        print(e)
    p_org.iat[i, 2] += 1
    print(p)
    p_org.to_csv("tg.csv", index=False)


bot.polling(non_stop=True,interval=0.1)
"""
https://cat-fact.herokuapp.com/fact
https://placekitten.com/200/100
GET /facts/random?animal_type=cat&amount=2
       
"""
""""""
"""bot.send_message(int(p[0]), "\n<i><b>" + text + "</b></i>", parse_mode="HTML")
bot.send_message(int(p[0]), "\n<i><b> IS FACT : " + str(j["status"]["verified"]).upper() + "</b></i>", parse_mode="HTML")"""
"""


def send_img(i):
    p_org = pd.read_csv("tg.csv")
    global vals_list
    print(p_org)
    v = len(p_org.values)
    url = f"https://cat-fact.herokuapp.com/facts/random?animal_type=cat&amount=1"
    response = requests.request("GET", url)
    print(response.text)
    j = json.loads(response.text)
    print(j)
    text = j["text"]

    p = p_org.iloc[i]
    print(j["status"]["verified"])
    with open("images.txt","r") as f:
        try:

            bot.send_photo(chat_id=int(p[0]),photo=f.read().split("\n")[p[2]])
            bot.send_message(int(p[0]),"HAPPY BIRTHDAY LOVE YOU")

        except Exception as e:
            print(e)
    p_org.iat[i, 2] += 1
    print(p)
    p_org.to_csv("tg.csv", index=False)


bot.polling()
"""
"""bot.send_photo(chat_id=int(p[0]),
                        photo=json.loads(requests.request("GET", "https://api.thecatapi.com/v1/images/search").text)[0][
                            "url"])"""
"""bot.send_message(int(p[0]), "\n<i><b>" + text + "</b></i>", parse_mode="HTML")
bot.send_message(int(p[0]), "\n<i><b> IS FACT : " + str(j["status"]["verified"]).upper() + "</b></i>", parse_mode="HTML")"""
