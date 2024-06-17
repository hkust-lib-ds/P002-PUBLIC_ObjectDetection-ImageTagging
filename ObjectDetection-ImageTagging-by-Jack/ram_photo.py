# This is one of the deliverables produced from this project: https://library.hkust.edu.hk/ds/project/p002/
# Created by LAU Ming Kit, Jack (Year 4 student, BEng in Computer Engineering, HKUST)

# Import libraries
import torch, requests, csv
import pandas as pd

# RAM
from PIL import Image, ImageFile
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

df = pd.read_csv('data.csv')

for i in range(6783, len(df)):
    try:
        # Global variable
        # Change this image path if needed
        image_pth = df.loc[i, 'id']
        response = requests.get("https://digitalimages.hkust.edu.hk/gallery/"+image_pth, stream=True)   

        # Prediction
        ori_image = Image.open(response.raw)

        transform = get_transform(image_size=image_size)
        with torch.no_grad():
            image = transform(ori_image).unsqueeze(0).to(device)

            res = inference(image, model)
        
        result = res[0].replace(' ', '').split('|')
        
        df.loc[i, 'keyword'] = '%s' % ','.join(map(str, result))
        
        print('processing:', i ,'/', len(df))
    except Exception as e:
        print('error at', i)
        print(e)
        df.to_csv('./db/data.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
        break
print('finished')
df.to_csv('./db/data.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
