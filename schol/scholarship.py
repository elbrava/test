import getpass
from bs4 import BeautifulSoup as Bs
import pyautogui
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import telebot

api_key = "5297152437:AAEYPa5YGkD-ZEYeqJ_ezqdo7DybwJaDUFc"
bot = telebot.TeleBot(api_key)
print(bot)
# Python code to add current script to the registry

# module to edit the windows registry
opt = Options()
opt.add_experimental_option("prefs", {
    "profile.default_content_setting_values.media_stream_mic": 1,

    "profile.default_content_setting_values.media_stream_camera": 1,

    "profile.default_content_setting_values.geolocation": 1,

    "profile.default_content_setting_values.notifications": 1

})
opt.add_argument("--disable-infobars")
opt.add_argument("--disable-extensions")
opt.add_argument("--use--fake-ui-for-media-stream")
opt.add_argument("--disable-gpu")
opt.add_argument("--no-sandbox")
opt.add_argument('--disable-dev-shm-usage')
chat = 1447290875
driver = webdriver.Chrome(chrome_options=opt, executable_path=r'chromedriver.exe')

spans = []

path = os.getcwd()
home = "https://www.wemakescholars.com"
driver.get('https://www.wemakescholars.com/scholarship/search?interest=11&country=168&study=2')

# document.getElementById("myCheck").click();
driver.implicitly_wait(700)
val = driver.find_element(By.CLASS_NAME, "records-found-num").text.split()[0]
print(val)
while True:
    p = len(driver.find_elements(By.CLASS_NAME, "post"))
    driver.execute_script("document.getElementById('load-more').click();")
    if p == int(val):
        break


def get_post():
    s = Bs(driver.page_source, 'lxml')
    # print(s.prettify())
    d = s.findAll("div", class_="post")
    count = 0
    for i in d:
        sp = i.findAll("span")
        hy = i.findAll("a")
        where = sp[5].text

        if sp[0].text != "Expired" and sp[2].text.strip() == "Full Funding":
            level = sp[1].getText().strip()
            output_string = f"""
NAME            :   {hy[2].text}
ELIGIBLE TO     :   {sp[4].text}
WHERE TO STUDY  :   {where}
COURSES         :   {sp[3].text}
LEVEL           :   {level}
EXPIRY          :   {sp[0].getText()} 
LINK            :   {home + hy[2]["href"]}
            """

            count += 1
            print(output_string)
    print(count)


get_post()
driver.quit()