# -*- coding: utf-8 -*-
import scrapy
from craigslist_rv.items import CraigslistRvItem #call items.py
from bs4 import BeautifulSoup as bs


class CraigslistBaseSpider(scrapy.Spider):
    name = "craigslist_base"
    allowed_domains = ["craigslist.org"]
    start_urls = (
        'https://losangeles.craigslist.org/search/sss?sort=rel&query=rvs',
    )

    def parse(self, response):
        titles = response.xpath("//a[@class='hdrlnk']/text()").extract()
        prices = response.xpath("//span[@class='price']/text()").extract()
        dates = response.xpath("//span[@class='pl']/time/@datetime").extract()
        
        for title, price, date in zip(titles, prices, dates):
        	item = CraigslistRvItem()
        	item["title"] = title
        	item["price"] = price.strip("$")
        	item["date"] = date
        	yield item 

        next_page_url = response.xpath("//span[@class='buttons']/a[@class='button next']/@href").extract_first()
        abs_next_page_url = response.urljoin(next_page_url)
        request = scrapy.Request(abs_next_page_url)
        yield request

    """

    def parse_job(self, response):
    	item = response.meta['item']
    	keys = ['title', 'price', 'date']
    	skills = ['title', 'price', 'date']

    	for key, skill in zip(keys, skills):
    		item[key] = int(skill in response.text)

    	yield item    

   """

  # response.xpath(*"//a[@class='hdrlnk']/text()"*).extract()