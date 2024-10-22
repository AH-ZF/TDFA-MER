###TDFA-MER

### requirements
opencv-contrib-python==4.9.0.80
pandas==2.2.1
pillow @ file:///croot/pillow_1707233021655/work
torch==2.3.0
torchaudio==2.3.0
torchstat==0.0.7
torchvision==0.17.2+cu121
xlrd==1.2.0
matplotlib==3.8.3
dlib==19.24.2
python-dateutil==2.9.0

Additional notes
For other environment requirements, please see the requirements.txt document under the path

### Dataset Preparation

Firstly,calculate u,v,os,maguv using preprocess.py

The dataset and txt file format are as follows:
*sub01
   **sub01/EP02_01f/u.jpg,sub01/EP02_01f/v.jpg,sub01/EP02_01f/os.jpg,sub01/EP02_01f/maguv.jpg,1

*sub02
   **……

……

###Best Model Parameters

Saved in ./result/modelpth



### Training and Testing the TDFA-MER

Training the 3-class data by using the following command:
```Bash  
cd “Code Path”
nohup python -u train_split.py > ./trainresult3C.log 2>&1 &
```

### Testing the best model for TDFA-MER

Testing the 3-class data by using the following command:
```Bash  
cd “Code Path”
nohup python -u last_test.py > ./trainresult3C.log 2>&1 &
```
