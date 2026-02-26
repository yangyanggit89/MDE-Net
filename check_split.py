import json
from collections import Counter

# Flickr8k
d = json.load(open('/root/project/dataset_flickr8k.json'))
print("=== Flickr8k ===")
print("Keys:", list(d.keys()))
imgs = d['images']
print("Total images:", len(imgs))
print("Sample keys:", list(imgs[0].keys()))
print("Count by split:", Counter(i['split'] for i in imgs))
print("Sample image:", imgs[0]['filename'])
print("Sample sentences:", imgs[0]['sentences'][0])

print("\n")

# Flickr30k
d = json.load(open('/root/project/dataset_flickr30k.json'))
print("=== Flickr30k ===")
print("Total images:", len(d['images']))
print("Count by split:", Counter(i['split'] for i in d['images']))
