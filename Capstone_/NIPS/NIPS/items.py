# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Item, Field

class NipsItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
	title = scrapy.Field()
	abstract = scrapy.Field()
	year = scrapy.Field()
