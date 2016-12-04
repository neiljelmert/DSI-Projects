# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider
from scrapy.selector import Selector
from scrapy.item import Item, Field
import urllib
from bs4 import BeautifulSoup as bs 
from CVPR_2016.items import Cvpr2016Item 
import requests

class CvprBaseSpider(scrapy.Spider):
	name = "CVPR_base"
	allowed_domains = ["cv-foundation.org"]
	start_urls = (
	'http://www.cv-foundation.org/openaccess/CVPR2013.py',
	)

	def parse(self, response):

		header = 'http://www.cv-foundation.org/openaccess/'
		posts = Selector(response).xpath("//dl/dt[@class='ptitle']/a/@href").extract()
		for post in posts:
			i = Cvpr2016Item()
			post = header + post
			
			request = scrapy.Request(post, callback = self.get_abs_title)
			request.meta['i'] = i
			yield request

	def get_abs_title(self, response):
		i = response.meta['i']
		titl = response.xpath("//dl/dd/div[@id='papertitle']").extract()[0]
		titl_soup  = bs(titl, 'html.parser')
		i['title'] = titl_soup.get_text().replace("\n","")
	
		abst = response.xpath("//dl/dd/div[@id='abstract']").extract()[0]
		abst_soup = bs(abst, 'html.parser')
		i['abstract'] = abst_soup.get_text().replace("\n","")

		yield i 