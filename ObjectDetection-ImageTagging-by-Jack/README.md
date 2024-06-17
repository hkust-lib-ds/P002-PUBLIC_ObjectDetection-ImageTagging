# RAM and GroundingDINO user guide

This is one of the deliverables produced from this project: https://library.hkust.edu.hk/ds/project/p002/
> Created by LAU Ming Kit, Jack (Year 4 student, BEng in Computer Engineering, HKUST)


## Before Installation
- Linux system is suggested when using this repository, window may cause error when downloading dependencies
- Official RAM repo: https://github.com/xinyu1205/recognize-anything
- Official GroundingDINO repo: https://github.com/IDEA-Research/GroundingDINO
- Official LLaVA repo: https://github.com/haotian-liu/LLaVA

## Installation guide

1. clone this repository using `git clone https://github.com/hkust-lib-ds/P002-PUBLIC_ObjectDetection-ImageTagging.git`
2. open the repository with terminal using
```
cd P002-PUBLIC_ObjectDetection-ImageTagging
cd ObjectDetection-ImageTagging-by-Jack
```
3. type `pip install requirement.txt` to install all the dependencies

OR via Jack's personal github:
1. clone this repository using `git clone https://github.com/Jacklau1216/image-tagging.git`
2. open the repository with terminal using `cd image-tagging`
3. type `pip install requirement.txt` to install all the dependencies

## RAM
- you can access the ram model using the jupyter notebook `ram_demo.ipynb`
- the following global variable can be changed:

|Variable| Meaning|
|--|--|
|`image_pth` | the file name that you want to recognize |

## GroundingDINO
- you can access the groundingDINO model using the jupter notebook `groundingDINO_demo.ipynb`
- the following global variable can be changed:

|Variable| Meaning|
|--|--|
|`image_pth` | the file name or folder that you want to recognize |
|`output_file`| the output directory|
|`TEXT_PROMPT`| object that you want to map (for single file)|

## LLaVA
- you can access the groundingDINO model using the jupter notebook `llava_demo.ipynb`
- the following global variable can be changed:

|Variable| Meaning|
|--|--|
|`image_file` | the file name that you want to recognize |
|`prompt`| the question you want to ask about the image|

