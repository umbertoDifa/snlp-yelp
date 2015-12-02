__author__ = 'vittorioselo'
import json

data =[]
dataPOS = []
with open('Edinburgh/ReviewsOfbusinessEdinburgh.json') as inputFile:
    data = json.load(inputFile)

with open('Edinburgh/ReviewsOfEdinburghPOS.json') as inputFile:
    dataPOS = json.load(inputFile)

print(data[1]['text'])
print(dataPOS[1]['text'])
