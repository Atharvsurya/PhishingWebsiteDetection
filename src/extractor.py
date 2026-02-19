import re
from urllib.parse import urlparse
import pandas as pd

def url_length(url):
    if len(url)>=52:
        return 1
    return 0

def has_at(url):
    if '@' in url:
        return 1
    return 0

def has_dash(url):
    if '-' in url:
        return 1
    return 0

def dot(url):
    if url.count('.')>2:
        return 1
    return 0

def no_of_subdomains(url):
    hostname = urlparse(url).netloc
    if hostname.count('.')>=3:
        return 1
    return 0

def has_ip(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',hostname):
        return 1
    return 0

data = pd.read_csv('data/raw_dataset.csv')
columns = pd.DataFrame({
    'url_length' : data['url'].apply(url_length),
    'has_at' : data['url'].apply(has_at),
    'has_dash' : data['url'].apply(has_dash),
    'dot' : data['url'].apply(dot),
    'has_ip': data['url'].apply(has_ip),
    'label' : data['label']
})
columns.to_csv('data/processed_dataset.csv',index=False)
print("Raw data is processed successfully...")