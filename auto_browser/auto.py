import webbrowser
from time import sleep

import selenium
import validators
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
inpu=""
drive = webdriver.Chrome("chromedriver.exe")
def var(topic):
    global inpu
    inpu = topic.replace(" ", "+")



def i_search(i):
    drive.implicitly_wait(4)
    drive.get("https://www.google.com/search?q=" + inpu + "&start" + str(i))
    drive.implicitly_wait(3)
    a = drive.find_element(By.XPATH, "//*[@id='hdtb-msb']/div[1]/div/div[2]/a")
    v = a.get_attribute("href")
    a.click()
    images = drive.find_elements(By.TAG_NAME, "img")

    images = [u.get_attribute("src") for u in images]
    images = [u for u in images if u != None]
    drive.implicitly_wait(3)

    _ = [print(u) for u in images]


def image_search():
    for i in range(1):
        try:
            i_search(i)
        except Exception as e:

            drive.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 't')
            # You can use (Keys.CONTROL + 't') on other OS
            i_search(i)


image_search()


def search_(i):
    drive.get("https://www.google.com/search?q=" + inpu + "&start" + str(i))
    drive.implicitly_wait(3)
    a = drive.find_elements(By.TAG_NAME, "a")
    a = [e.get_attribute("href") for e in a]
    a = [e for e in a if e != None]
    print(len(a))
    _ = [print(e) for e in a if not e.__contains__("google")]
    print(len(a))


def search():
    for i in range(1):
        print("called")
        search_(i)

        # key elements

        # enjoy


search()

drive.implicitly_wait(300)
# p = drive.find_element_by_name("a")
# article links
# pictures
# print(p)
drive.quit()
