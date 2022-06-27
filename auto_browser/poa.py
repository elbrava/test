import webbrowser
from time import sleep

import requests
import selenium
import validators
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


def i_search(link):
    r = requests.get(link)
    l = len(r.content)

    if l == 2864 or l == 0:
        print(l)
        raise ValueError("IN")
    else:
        print(r.content)


# 10.1.0.254

for i in range(10,255):
    for o in range(255):
        for p in range(255):
            for a in range(255):
                ip = f"{i}.{o}.{p}.{a}"
                link = fr"https://uko.poa.im/hotspotlogin.php?res=notyet_mk&mac=B6:D6:8E:8B:58:A6&nasid=08-55-31-A5\
                -4F-E6&ip={ip} " + r"&linklogin=http://10.1.0.1/login&userurl=http://www.msftconnecttest.com/redirect" + r"&logouturl=http://10.1.0.1" + r"/logout&statusurl=http://10.1.0.1/status&servername=08-55-31-A5-4F-E6&identity=D11B0DE466F1&apname=10.1.0.1" + r"&serveraddress=10.1.0.1:80 "

                print(link)
                try:
                    i_search(link)
                except Exception as e:
                    print(e)
                else:
                    exit()
