import re
from urllib.parse import urlparse
import pandas as pd

## 1 -> Phishing /// 0 -> Legitimate 

shortners = pd.read_csv('data/shortners.csv')

# Address Bar Based Features

def url_length(url):
    if len(url)>=54:
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
    hostname = urlparse(url).netloc
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',hostname):
        return 1
    return 0

def shortening_service(url):
    return 1 if any(s in url for s in shortners) else 0

def double_slash_redirect(url):
    return 1 if url.count('//') > 1 else 0

def ssl_final_state(url):
    return int(not url.startswith("https://"))

def https_token(url):
    hostname = urlparse(url).netloc.lower()
    return int("https" in hostname)

def redirect_symbol(url):
    return int('//' in url[8:])

data = pd.read_csv('data/raw_dataset.csv')
columns = pd.DataFrame({
    'url_length' : data['url'].apply(url_length),
    'has_at' : data['url'].apply(has_at),
    'has_dash' : data['url'].apply(has_dash),
    'dot' : data['url'].apply(dot),
    'has_ip': data['url'].apply(has_ip),
    'shortening_service' : data['url'].apply(shortening_service),
    'double_slash_redirect' : data['url'].apply(double_slash_redirect),
    'ssl_final_state' : data['url'].apply(ssl_final_state),
    'https_token' : data['url'].apply(https_token),
    'redirect_symbol' : data['url'].apply(redirect_symbol),
    'label' : data['label']
})

columns.to_csv('data/processed_dataset_v2.csv',index=False)
print("Raw data is processed successfully...")