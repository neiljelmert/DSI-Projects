# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy import Item, Field


class RedditItem(scrapy.Item):
	# # define the fields for your item here like:
	title = scrapy.Field()
	url = scrapy.Field()
	submitted = scrapy.Field()
	link = scrapy.Field()
	abstract = scrapy.Field()
	top_comment = scrapy.Field()
	votes = scrapy.Field()
	rank = scrapy.Field()
