	# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider
from scrapy.selector import Selector
from scrapy.item import Item, Field
import urllib
from bs4 import BeautifulSoup as bs 
from NIPS.items import NipsItem 
import requests


class NipsBaseSpider(scrapy.Spider):
	name = "NIPS_base"
	allowed_domains = ["papers.nips.cc"]
	start_urls = ['http://papers.nips.cc/book/advances-in-neural-information-processing-systems-29-2016',
				'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-28-2015',
				'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-27-2014',
				'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-26-2013',
				'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012',
				'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-24-2011',
				'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-23-2010',
				'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-22-2009', 
				'http://papers.nips.cc/book/advances-in-neural-information-processing-systems-21-2008']
	custom_settings = {'DOWNLOAD_DELAY': 2}

	def parse(self, response):

		header = 'http://papers.nips.cc/'
		posts = Selector(response).xpath("//div[@class='main wrapper clearfix']/ul//a[1]/@href").extract()
		for post in posts:
			i = NipsItem()
			post = header + post
			
			request = scrapy.Request(post, callback = self.get_abs_title)
			request.meta['i'] = i
			yield request

	def get_abs_title(self, response):
		i = response.meta['i']
		titl = response.xpath("//div[@class='main wrapper clearfix']/h2[@class='subtitle']").extract()[0]
		titl_soup  = bs(titl, 'html.parser')
		i['title'] = titl_soup.get_text().replace("\n","")
	
		abst = response.xpath("//div[@class='main wrapper clearfix']/p[@class='abstract']").extract()[0]
		abst_soup = bs(abst, 'html.parser')
		i['abstract'] = abst_soup.get_text().replace("\n","")

		year = response.xpath("//header[@class='wrapper clearfix']/nav/ul/li[2]/a").extract()[0]
		year_soup = bs(year, 'html.parser')
		i['year'] = year_soup.get_text()

		yield i 










