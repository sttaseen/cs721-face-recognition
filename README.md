# cs721-face-recognition

Part 1 for project 1 for COMPSYS721: Special Topic


## Setup

### Create and Activate Conda Environment

```bash
conda create -n face python=3.8 -y
conda activate face
```
### Install Dependencies
1. Install requirements from requirements.txt:

```
pip install -r requirements.txt
```

2. Install PyTorch:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. [Optional] Jupyter Notebook Setup

```bash
conda install ipykernel -y
ipython kernel install --user --name=face
```

### Set up Dataset
Extract LFW dataset into data folder. The directory should look like this:
```
.
├── data
│   ├── lfw
│   │   ├── Ariel_Sharon
│   │   │   ├── Ariel_Sharon_0001.jpg
│   │   │   ├── Ariel_Sharon_0002.jpg
│   │   │   └── ..
│   │   ├── ..
│   ├── lfw.tgz
│   └── README.txt
├── evaluation
│   ├── efficientnet-b2.ipynb
│   ├── inception-v1.ipynb
│   └── resnet18.ipynb
├── LICENSE
├── preprocessing
│   └── preprocessing.ipynb
├── project_brief.pdf
├── README.md
├── requirements.txt
├── tuning
│   ├── efficientnetb2-gridsearch.ipynb
│   ├── inceptionv1-gridsearchcv.ipynb
│   ├── resnet18-gridsearchcv.ipynb
│   └── utils.py
└── visualizations
    └── ..
```


## Notes

1. First, run the notebook in `/preprocessing`.
2. Tune the models using notebooks in `/tuning`.
3. Evaluate using the notebooks in `/evaluation`.
