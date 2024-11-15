import scrapy


class SimpleSpider(scrapy.Spider):
    name = 'oic'
    start_urls = ["https://docs.oracle.com/en/cloud/paas/integration-cloud/books.html"]

    def parse(self, response):
        pdf_links = response.xpath('//a[contains(@href, ".pdf") and @aria-labelledby]')
        counter = 1
        
        for link in pdf_links:
            pdf_url = response.urljoin(link.xpath('@href').get())
            title = str(counter)
        
            counter += 1
            
            self.log(f'PDF URL: {pdf_url}, Title: {title}')
            item = {
                'Title': title,
                'file_urls': [pdf_url]
            }
            yield item
