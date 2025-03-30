# Face Segmentatoin using BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

## Installation

To get started with the Face Parsing Model, clone this repository and install the required dependencies:

```commandline
git clone https://github.com/Saran3072/face-segmentation.git
cd face-segmentation
pythin3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

#### Download weights (click to download):

| Model    | PT                                                                                         
| -------- | ------------------------------------------------------------------------------------------
| ResNet18 | [resnet18.pt](https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet18.pt)
| ResNet34 | [resnet34.pt](https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet34.pt)

### Run streamlit app
```commandline
streamlit run app.py
