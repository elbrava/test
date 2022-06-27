import json
import os
import pathlib
import threading

import requests
import telebot
import pandas as pd
import time
import sys

from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

sys.setrecursionlimit(100000000)
time_begin = time.perf_counter()
api_key = "5211132462:AAGEQo5ueNYnnXVrxz7kN0wtoCEGvnr-1lM"
# bot = telebot.TeleBot(api_key)
# print(bot)

vals_list = []
inpu = "love romantic beautiful art".replace(" ", "+")
drive = webdriver.Chrome("chromedriver.exe")


def i_search(i):
    drive.implicitly_wait(4)
    try:
        drive.get("https://www.google.com/search?q=" + inpu + "&start" + str(i))
        drive.implicitly_wait(3)

        a = drive.find_element(By.XPATH, "//*[@id='hdtb-msb']/div[1]/div/div[2]/a")
        v = a.get_attribute("href")
    except:
        drive.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 't')
        drive.get("https://www.google.com/search?q=" + inpu + "&start" + str(i))
        drive.implicitly_wait(3)

        a = drive.find_element(By.XPATH, "//*[@id='hdtb-msb']/div[1]/div/div[2]/a")
        v = a.get_attribute("href")
    a.click()
    SCROLL_PAUSE_TIME = 4

    # Get scroll height
    last_height = drive.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        drive.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = drive.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        images = drive.find_elements(By.TAG_NAME, "img")
        last_height = new_height
    images = drive.find_elements(By.TAG_NAME, "img")

    images = [u.get_attribute("src") for u in images]
    images = [u for u in images if u != None]
    drive.implicitly_wait(3)

    _ = [save(u) for u in images]


def save(u):
    with open("images.txt", "a") as f:
        f.write(u)
        f.write("\n")


def image_search():
    i_search(1)


# You can use (Keys.CONTROL + 't') on other OS


image_search()
drive.quit()
