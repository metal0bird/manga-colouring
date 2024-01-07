import urllib.request
import urllib
import json
import numpy as np
import cv2
import untangle
import os
import matplotlib.pyplot as plt
import ssl
import xml.etree.ElementTree as ET
ssl._create_default_https_context = ssl._create_unverified_context

# The API template for pulls is given by Safebooru https://safebooru.org/index.php?page=help&topic=dapi
# /index.php?page=dapi&s=post&q=index
maxsize = 512
imagecounter = 0
maxImages = 10000
pagestepper = 0
pageoffset = 50
tags = 'orange_eyes'
savepath = 'Images_orange_eyes/'

while imagecounter < maxImages:
    #Get a taged page

    url = f"http://safebooru.org/index.php?page=dapi&s=post&q=index&tags={tags}&pid={pageoffset + pagestepper}"

    pagestepper += 1

    xmlreturn = untangle.parse(url)
    for post in xmlreturn.posts.post:
        imgurl = post["sample_url"]
        if ("png" in imgurl) or ("jpg" in imgurl):
            resp = urllib.request.urlopen(imgurl)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if isinstance(image, type(None)):
                # this happens when we run into 404 error on website
                continue
            print('counter: {}. URL: {}'.format(imagecounter, imgurl))

            height, width = image.shape[:2]
            # Resize but preserve aspect ratio-- this requires we crop and cut the image blindly!
            if height > width:
                    scalefactor = maxsize / width
                    resized = cv2.resize(
                            image,
                            (int(width * scalefactor), int(height * scalefactor)),
                            interpolation=cv2.INTER_CUBIC
                    )
                    cropped = resized[:maxsize, :maxsize]
            else:
                    scalefactor = maxsize / height
                    resized = cv2.resize(
                            image,
                            (int(width * scalefactor), int(height * scalefactor)),
                            interpolation=cv2.INTER_CUBIC
                    )
                    center_x = int(round(width * scalefactor * 0.5))
                    cropped = resized[:maxsize, center_x - maxsize // 2:center_x + maxsize // 2]

            # so we now have resized/cropped image pulled from the website
            cv2.imwrite(
                savepath + '/' + str(imagecounter) + '_page_' +
                str(pagestepper + pageoffset) + ".jpg", cropped)
            if imagecounter == maxImages:
                break
            else:
                imagecounter += 1





