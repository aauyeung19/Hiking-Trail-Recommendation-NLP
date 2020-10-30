"""
@Andrew

This is a script for scraping hiking trail information from All Trails.
username: andrew.k.auyeung@gmail.com
Alltrails Pw: 8FEcfxEb/%eyVQ7
"""
from bs4 import BeautifulSoup
from selenium import webdriver
import pickle
import time



browser = webdriver.Chrome(executable_path='/Applications/chromedriver')
browser.get('http://www.alltrails.com')
to_login = browser.find_element_by_xpath("//div[@data-test-id='navigation-loginButton']")
to_login.click()
# maybe sleep
username = browser.find_element_by_name('userEmail')
password = browser.find_element_by_name('userPassword')
username.send_keys('andrew.k.auyeung@gmail.com')
password.send_keys("8FEcfxEb/%eyVQ7")
browser.find_element_by_xpath("//input[@data-test-id='formButton-submit']").click()