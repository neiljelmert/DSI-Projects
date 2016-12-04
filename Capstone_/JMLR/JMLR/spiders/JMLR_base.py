	# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider
from scrapy.selector import Selector
from scrapy.item import Item, Field
import urllib
from bs4 import BeautifulSoup as bs 
from JMLR.items import JmlrItem 
import requests


class JmlrBaseSpider(scrapy.Spider):
	name = "JMLR_base"
	allowed_domains = ["jmlr.org"]
	start_urls = ['http://jmlr.org/proceedings/papers/v28/'] # v48 = 16, v32 = 14, v37 = 15, v28 = 13

	def parse(self, response):

		header = 'http://jmlr.org/proceedings/papers/v28/'
		posts = Selector(response).xpath("//p[@class='links']/a[1]/@href").extract()

		for post in posts:
			i = JmlrItem()
			post = header + post

			i['year'] = 2013

			request = scrapy.Request(post, callback = self.get_abs_title)
			request.meta['i'] = i
			yield request

	def get_abs_title(self, response):

		i = response.meta['i']
		titl = response.xpath("//div[@id='content']/h1").extract()[0]
		titl_soup  = bs(titl, 'html.parser')
		i['title'] = titl_soup.get_text().replace("\n","")
	
		abst = response.xpath("//div[@id='content']/div[@id='abstract']").extract()[0]
		abst_soup = bs(abst, 'html.parser')
		i['abstract'] = abst_soup.get_text().replace("\n","")


		yield i 
