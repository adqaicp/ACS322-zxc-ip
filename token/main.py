# encoding:utf-8

import requests
import base64

'''
手写文字识别
'''

request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting"
# 二进制方式打开图片文件
f = open('I:\\PYTHON\\token\\checck\\丘.png', 'rb')
img = base64.b64encode(f.read())

params = {"image":img}
access_token = 'your token'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    print (response.json())