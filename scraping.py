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
import psycopg2




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
    # maybe sleep
    username = browser.find_element_by_name('userEmail')
    password = browser.find_element_by_name('userPassword')
    username.send_keys('andrew.k.auyeung@gmail.com')
    password.send_keys("8FEcfxEb/%eyVQ7")
    browser.find_element_by_xpath("//input[@data-test-id='formButton-submit']").click()
    time.sleep(5)
    browser.get('https://www.alltrails.com/us/new-jersey')
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
            time.sleep(1)
        except:
            break
    return BeautifulSoup(browser.page_source)

def get_all_reviews(link):
    while True:
        try:
            browser.find_element_by_xpath("//button[@title='Show more reviews']").click()
            time.sleep(1)
        except:
            break
    return BeautifulSoup(browser.page_source)


def parse_hike(soup, hike_numb):
    """
    Parse Soup from Hike Page
    """
    hike_id = 'hike_' + str(hike_numb)
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
        "trail_tags": tags}
    
    return hike_row, reviews

def to_db(hike_row, reviews, conn)

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

if __name__=="__main__":
    browser = webdriver.Chrome(executable_path='/Applications/chromedriver')
    browser = login(browser)
    for n, link
        hike_row, reviews = parse_hike(soup, n)
        conn=psycopg2.connect(database='alltrails', user='postgres', host='127.0.0.1', port= '5432')
        to_db(hike_row, reviews, conn)