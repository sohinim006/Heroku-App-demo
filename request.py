import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'state':1,'year':2014,'npH':80,'ndo':100,'nec':60,'nbdo':100,'nna':100})

print(r.json())