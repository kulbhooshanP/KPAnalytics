import pandas as pd
from pytrends.request import TrendReq
pytrend = TrendReq(hl='en-US', tz=330)
keywords = ['Vodafone', 'JIO', 'Airtel']
pytrend.build_payload(
     kw_list=keywords,
     cat=13,
     timeframe='today 12-m',
     geo='IN',
     gprop='')
data = pytrend.interest_over_time()
data= data.drop(labels=['isPartial'],axis='columns')
image = data.plot(title = 'Telecom Trends in last 3 months on Google Trends ')
fig = image.get_figure()
fig.savefig('figure.png')
data.to_csv('Telecom Trends.csv', encoding='utf_8_sig')
pytrend.related_topics()

pytrend.trending_searches(pn='india') # trending searches in real time for United States
pytrend.suggestions('airtel')
pytrend.categories()
x=pytrend.related_queries()
pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
