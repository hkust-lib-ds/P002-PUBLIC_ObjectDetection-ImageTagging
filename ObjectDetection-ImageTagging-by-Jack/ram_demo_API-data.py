# This is one of the deliverables produced from this project: https://library.hkust.edu.hk/ds/project/p002/
# Created by LAU Ming Kit, Jack (Year 4 student, BEng in Computer Engineering, HKUST)

import requests
import xml.etree.ElementTree as ET
import pandas as pd
# Import libraries
import torch

# RAM
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from io import BytesIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

url='https://lbezone.hkust.edu.hk/rse/OAI/Server.php?verb=ListRecords&metadataPrefix=oai_dc&set=shafei'
response = requests.get(url)
response.encoding = 'utf-8'
xml_content = response.content

# Parse the XML content
root = ET.fromstring(xml_content)

# Create lists to store the extracted information
titles = []
dates = []
languages = []
urls = []
thumb_urls =[]

# Iterate over each record and extract the desired information
for record in root.findall(".//{http://www.openarchives.org/OAI/2.0/}record"):
    title = record.find(".//{http://purl.org/dc/elements/1.1/}title").text
    date = record.find(".//{http://purl.org/dc/elements/1.1/}date").text

    titles.append(title)
    dates.append(date)

    identifiers = record.findall('.//{http://purl.org/dc/elements/1.1/}identifier')
    thumb_url = identifiers[1].text
    thumb_urls.append(thumb_url)

    doi_url = identifiers[4].text
    urls.append(doi_url)
    
# Create a dataframe from the extracted information
data = {
    "Title": titles,
    "Year": dates,
    "URL": urls,
    "Thumbnail image": thumb_urls
}
data = pd.DataFrame(data)

# Prediction
# download the model from the URL below and put in the "pretrained" folder of your working directory 
# https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth
ram_pth =  "pretrained/ram_plus_swin_large_14m.pth"
image_size = 384

#######load model
model = ram_plus(pretrained=ram_pth,
                            image_size=image_size,
                            vit='swin_l')
model.eval()

model = model.to(device)

for i in range(len(data)):
    response = requests.get(data.loc[i, 'Thumbnail image'])
    ori_image = Image.open(BytesIO(response.content))

    transform = get_transform(image_size=image_size)

    image = transform(ori_image).unsqueeze(0).to(device)

    res = inference(image, model)
    
    data.loc[i, 'keywords'] = res[0]
    
data.to_csv('output/websiteAPI.csv')
