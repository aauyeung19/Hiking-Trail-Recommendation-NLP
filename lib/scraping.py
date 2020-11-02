"""
@Andrew

This is a script for scraping hiking trail information from All Trails.
username: andrew.k.auyeung@gmail.com
Alltrails Pw: 8FEcfxEb/%eyVQ7
"""
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import psycopg2
import pickle
import numpy as np

def login(browser):
    """
    Logs into AllTrails 
    args:
        browser: Webdriver Default- Chrome
    returns:
        browser
    """
    browser.get('http://www.alltrails.com')
    to_login = browser.find_element_by_xpath("//div[@data-test-id='navigation-loginButton']")
    to_login.click()
    time.sleep(3)
    username = browser.find_element_by_name('userEmail')
    password = browser.find_element_by_name('userPassword')
    username.send_keys('andrew.k.auyeung@gmail.com')
    password.send_keys("8FEcfxEb/%eyVQ7")
    time.sleep(2)
    browser.find_element_by_xpath("//input[@data-test-id='formButton-submit']").click()
    time.sleep(5)
    browser.get('https://www.alltrails.com/us/pennsylvania')
    return browser

def get_all_hikes(browser):
    """
    Clicks through main page of All Trails
    Gets links to all trails on page

    returns: 
        soup: BeautifulSoup object of the webpage. 
    """
    while True:
        try:
            browser.find_element_by_xpath("//button[@title='Show more trails']").click()
            time.sleep(2*np.random.rand())
            all_hikes_soup = BeautifulSoup(browser.page_source, 'lxml')
            if len(all_hikes_soup.find_all("a", {'itemprop': 'url'}))==1000:
                break
        except:
            break
    return BeautifulSoup(browser.page_source, 'lxml')

def get_all_reviews(link, browser):
    browser.get(link)
    stale = 0
    while True:
        try:
            soup = BeautifulSoup(browser.page_source, 'lxml')
            rows = soup.find('div', class_='feed-items null').find_all('div', {'itemprop':'review'})
            curr_len = len(rows)
            browser.find_element_by_xpath("//button[@title='Show more reviews']").click()
            time.sleep(3*np.random.rand())
            soup = BeautifulSoup(browser.page_source, 'lxml')
            rows = soup.find('div', class_='feed-items null').find_all('div', {'itemprop':'review'})
            if curr_len == len(rows):
                stale+=1
                time.sleep(7)
            if stale == 5:
                break
        except:
            break
    return BeautifulSoup(browser.page_source, 'lxml')


def parse_hike(soup, hike_numb, link):
    """
    Parse Soup from Hike Page
    """
    hike_id = 'hike_' + str(hike_numb+4001)
    state = soup.find('ul', id='breadcrumb-list').find_all('span', {"itemprop":"name"})[1].text
    title = soup.find('ul', id='breadcrumb-list').find_all('span', {"itemprop":"name"})[3].text
    forest = soup.find('ul', id='breadcrumb-list').find_all('span', {"itemprop":"name"})[2].text
    description = soup.find(id="trail-top-overview-text").find('p').text
    try:
        length = soup.find(text='Length').next.next.text.strip()
    except:
        length = None
    try:
        elevation = soup.find(text='Elevation gain').next.next.text.strip()
    except:
        elevation = None
    try:
        r_type = soup.find(text='Route type').next.next.text.strip()
    except:
        r_type = None

    # tags
    tags = []
    for each in soup.find('section', class_='tag-cloud').find_all('span', class_="big rounded active"):
        tags.append(each.text)
    
    # Get user Ratings
    reviews = []
    rows = soup.find('div', class_='feed-items null').find_all('div', {'itemprop':'review'})
    for row in rows:
        user_id = row.find('a')['href'].replace('/members/', '')
        user_rating = row.find_all('span')[0]['aria-label']
        date = row.find_all('span')[0].nextSibling.text
        try:
            user_desc = row.find('p', {'itemprop': 'reviewBody'}).text
        except:
            user_desc = None

        reviews.append(
            {'hike_id': hike_id,
            'user_id': user_id,
            'date': date,
            'user_desc': user_desc,
            'user_rating': user_rating})
    
    hike_row = {"hike_id": hike_id,
        "trail_name": title,
        "state": state,
        "park": forest,
        "trail_description": description,
        "trail_length": length,
        "trail_elevation": elevation,
        "trail_type": r_type,
        "trail_tags": tags,
        "link": link}
    
    return hike_row, reviews

def to_db(hike_, review_, conn):
    """
    Puts hike information into hikes database
    and reveiw information into reviews database
    """
    columns_ = ','.join(hike_)
    val = ', '.join(["%s"]*len(hike_))
    query = f"INSERT INTO hikes ({columns_}) values ({val});"

    cursor = conn.cursor()
    cursor.execute("BEGIN;")
    cursor.execute(query, list(hike_.values()))
    cursor.execute("commit;")


    columns_ = ','.join(review_[0])
    val = ', '.join(["%s"]*len(review_[0]))
    query = f"INSERT INTO reviews ({columns_}) values ({val});"
    for each in review_:
        cursor = conn.cursor()
        cursor.execute("BEGIN;")
        cursor.execute(query, list(each.values()))
        cursor.execute("commit;")

def upload_to_sql(link_stubs, start=0):
    base_url = "https://www.alltrails.com"
    for n, link in enumerate(link_stubs[start:], start+1):
        url = base_url+link['href']
        hike_soup = get_all_reviews(url, browser)

        hike_row, reviews = parse_hike(hike_soup, n, url)
        conn=psycopg2.connect(database='alltrails', user='postgres', host='127.0.0.1', port= '5432')
        to_db(hike_row, reviews, conn)


if __name__=="__main__":
    browser = webdriver.Chrome(executable_path='/Applications/chromedriver')
    browser = login(browser)
    all_hikes_soup = get_all_hikes(browser)

    link_stubs = all_hikes_soup.find_all("a", {'itemprop': 'url'})
    upload_to_sql(link_stubs, start=0)
    browser.quit()
