import requests
from bs4 import BeautifulSoup as bs 
import pickle
import time
import csv

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

forum_idsICLR16 = pickle.load(open('/Users/ga/Desktop/Capstone/forumids_for_openreviewICLR16', 'rb'))

# forum_ids = '/forum?id=rJc0or11l/forum?id=HyiC6J1Je/forum?id=Sk6W5JJke/forum?id=BJnqmRAA/forum?id=BkfmIt0R'.split("/")
# forum_idsNIPS16_NAMPI = forum_ids[1:]

# forum_ids = '/forum?id=r1oidEgJl/forum?id=ByjDCQgke/forum?id=rk4pPfxJx/forum?id=SkvrtAJJe/forum?id=S1uHiFyyg/forum?id=H1-FoHkJx/forum?id=Hy9H-GAC/forum?id=SkgbhoyC'.split("/")
# forum_idsNIPS16_MLITS = forum_ids[1:]

open_rev_url = 'https://openreview.net/'

req = requests.Session()

# to get `GCLB` cookie
r = req.get(open_rev_url)

# to get `openreview:sid` cookie
r = req.get('http://openreview.net/token')


def get_abstract(fid):
	time.sleep(0.3)
	r = req.get('http://openreview.net/notes?forum=' + str(fid[9:]) + '&trash=true')
	j = r.json()
	for note in j['notes']:
		if 'abstract' in note['content'].keys():
			abstract = note['content']['abstract']
	return abstract 

def get_title(fid):
	time.sleep(0.3)
	r = req.get('http://openreview.net/notes?forum=' + str(fid[9:]) + '&trash=true')
	j = r.json()
	for note in j['notes']:
		if 'title' in note['content'].keys():
			title = note['content']['title']
	return title 


alldata = [[get_title(fid), get_abstract(fid)] for fid in forum_idsICLR16]

with open('open_review_abstracts_ICLR16.csv', 'wb') as f:
	writer = csv.writer(f)
	writer.writerow(["title", "abstract"])
	writer.writerows(alldata)
