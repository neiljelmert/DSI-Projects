

import scrapy
from scrapy.spiders import CrawlSpider
from scrapy.selector import Selector
from scrapy.item import Item, Field
import urllib
from bs4 import BeautifulSoup
from reddit.items import RedditItem 
import requests

class RedditCrawler(scrapy.Spider):
	name = 'reddit'
	allowed_domains = ['reddit.com', 'openreview.net'] #arxiv.org for when arxiv scraping 
	start_urls = ['https://www.reddit.com/r/MachineLearning']
	custom_settings = {
	'BOT_NAME': 'reddit',
	#'DEPTH_LIMIT': 5,
	'DOWNLOAD_DELAY': 2
	}


	def parse(self, response):
		s = Selector(response)
		next_page = s.xpath('.//span[@class="next-button"]/a/@href').extract()
		if next_page:
			next_href = next_page[0]
			request = scrapy.Request(url=next_href)
			yield request

		


###################################################################################
# OPENREVIEW

	# def parse(self, response):
	# 	s = Selector(response)
	# 	next_page = s.xpath('.//span[@class="next-button"]/a/@href').extract()
	# 	if next_page:
	# 		next_href = next_page[0]
	# 		request = scrapy.Request(url=next_href)
	# 		yield request
	
	# 	posts = Selector(response).xpath('//div[@id="siteTable"]/div[@onclick="click_thing(this)"]')

	# 	open_rev_posts = [post for post in posts if post.xpath('div[2]/p[1]/span[@class="domain"]/a/@href').extract()[0] == '/domain/openreview.net/']
		
	# 	for post in open_rev_posts:
	# 		i = RedditItem()
			
	# 		open_rev_url = post.xpath('.//p[@class="title"]/a/@href').extract()[0]

	# 		i['title'] = post.xpath(
	# 				'div[2]/p[1]/a/text()').extract()[0]
	# 		i['url'] = post.xpath(
	# 				'div[2]/ul/li[1]/a/@href').extract()[0]
	# 		i['submitted'] = post.xpath(
	# 				'div[2]/p[2]/time/@title').extract()[0]
	# 		i["votes"] = post.xpath(
	# 			'.//div[@class="midcol unvoted"]/div[@class="score unvoted"]/text()').extract()[0]
	# 		i["rank"] = post.xpath(
	# 			'.//span[@class="rank"]/text()').extract()[0]

	# 		i['link'] = open_rev_url


	# 		if "pdf" in open_rev_url:
	# 			open_review_id = open_rev_url.split('=')[1]
	# 			open_rev_url = 'http://openreview.net/forum?id=' + open_review_id
	# 			i['link'] = open_rev_url

	# 			req = requests.Session()

	# 			# to get `GCLB` cookie
	# 			r = req.get(open_rev_url)

	# 			# to get `openreview:sid` cookie
	# 			r = req.get('http://openreview.net/token')

	# 			# to get JSON data
	# 			r = req.get('http://openreview.net/notes?forum=' + open_review_id + '&trash=true')
	# 			j = r.json()

	# 			abstract_open_review = []
	# 			for note in j['notes']:
	# 				if "abstract" in note['content'].keys():
	# 					abstract_open_review.append(note['content']['abstract'])

	# 			i["abstract"] = " ".join(abstract_open_review)

		
	# 		elif "pdf" not in open_rev_url:
	# 			# try:
	# 			open_review_id = open_rev_url.split('=')[1]
	# 			i['link'] = open_rev_url

	# 			req = requests.Session()

	# 			# to get `GCLB` cookie
	# 			r = req.get(open_rev_url)

	# 			# to get `openreview:sid` cookie
	# 			r = req.get('http://openreview.net/token')

	# 			# to get JSON data'
	# 			r = req.get('http://openreview.net/notes?forum=' + open_review_id + '&trash=true')
	# 			j = r.json()

	# 			abstract_open_review = []
	# 			for note in j['notes']:
	# 				if "abstract" in note['content'].keys():
	# 					abstract_open_review.append(note['content']['abstract'])

	# 			i["abstract"] = " ".join(abstract_open_review)


	# 		yield i 

#####################################################################################################
# ARXIV


	# def parse(self, response):
	# 	s = Selector(response)
	# 	next_page = s.xpath('.//span[@class="next-button"]/a/@href').extract()
	# 	if next_page:
	# 		next_href = next_page[0]
	# 		request = scrapy.Request(url=next_href)
	# 		yield request
	

	# 	posts = Selector(response).xpath('//div[@id="siteTable"]/div[@onclick="click_thing(this)"]')
		
		#arx_posts = [post for post in posts if post.xpath('div[2]/p[1]/span[@class="domain"]/a/@href').extract()[0] == '/domain/arxiv.org/']

	# 	for post in arx_posts:
	# 		i = RedditItem()
			
	# 		arx_url = post.xpath('.//p[@class="title"]/a/@href').extract()[0]

	# 		i['title'] = post.xpath(
	# 				'div[2]/p[1]/a/text()').extract()[0]
	# 		i['url'] = post.xpath(
	# 				'div[2]/ul/li[1]/a/@href').extract()[0]
	# 		i['submitted'] = post.xpath(
	# 				'div[2]/p[2]/time/@title').extract()[0]
	# 		i["votes"] = post.xpath(
	# 			'.//div[@class="midcol unvoted"]/div[@class="score unvoted"]/text()').extract()[0]
	# 		i["rank"] = post.xpath(
	# 			'.//span[@class="rank"]/text()').extract()[0]

	# 		i['link'] = arx_url


	# 		if "pdf" not in arx_url:
	# 			request = scrapy.Request(arx_url, callback = self.get_arxiv_abstract)
	# 			print request
	# 			request.meta['i'] = i
	# 			yield request

	# 			# yield i
			
	# 		elif "pdf" in arx_url:
	# 			head = arx_url[:18]
	# 			slash = arx_url.split("/")
	# 			id_and_pdf = slash[-1]
	# 			arxiv_id = id_and_pdf[:-4]
	# 			arx_url_new = head + "abs/" + arxiv_id
	# 			print arx_url_new

	# 			request = scrapy.Request(arx_url_new, callback = self.get_arxiv_abstract)
	# 			request.meta['i'] = i
	# 			yield request

	# 		comments_url = post.xpath('.//li[@class="first"]/a/@href').extract()[0]
	# 		com_request = scrapy.Request(comments_url, callback = self.get_top_comment)
	# 		com_request.meta['i'] = i
	# 		yield com_request

				
	# def get_top_comment(self, response):
	# 	i = response.meta['i']
	# 	try:
	# 		top = response.xpath('.//div[@class="commentarea"]//div[@class="md"]').extract()[0]
	# 		top_soup = BeautifulSoup(top, 'html.parser')
	# 		i["top_comment"] = top_soup.get_text().replace('\n', ' ')
	# 	except IndexError:
	# 		i["top_comment"] = 0

	# def get_arxiv_abstract(self, response):
	# 	i = response.meta['i']
	# 	try:
	# 		abstract_clean = response.xpath(".//blockquote[@class='abstract mathjax']").extract()[0]
	# 		abstract_soup = BeautifulSoup(abstract_clean, 'html.parser')
	# 		i['abstract'] = abstract_soup.get_text().replace('\n', ' ')
	# 	except AttributeError:
	# 		i["abstract"] = 0

	#  	yield i 












