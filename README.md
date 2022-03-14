# Arabic_Dialect_Identification_NLP-AIM-Task
 ### Arabic has a widely varying collection of dialects. Many of these dialects remain under-studied due to the rarity of resources. 

The goal of AIM task is to classifying the dialect of the tweet writer given the tweet itself.

#### I used two approches:
##### 1- Machine Learning Aproach Using `LinearSVC`
##### 2- Deep Learning Approach Using `AraBERT Transformer`

### Deep Learning AraBERT Fine Tuning:
###### I fine tuned `AraBERTv0.2-Twitter` base/large you found in <a href="https://huggingface.co/aubmindlab/bert-base-arabertv02-twitter">this link</a> and the models weights in <a href="https://drive.google.com/drive/folders/1-SUcZCJbWmP9Mf8vI4XMFR0fv5ruDmdm?usp=sharing">this drive link</a>

###### To use AraBERT Models go to the NoteBook Folder or <a href="https://colab.research.google.com/drive/1a7kSCVrwHMPFA2BfaBz3FjsmnLJXtfZ2?usp=sharing">This CoLab Link</a>
-------------------------------------------------------------------------------------------
### The Deployment is by Machine learning model and steps to run the flask
##### 1- Download the pkl model from the drive in <a href="https://drive.google.com/file/d/10rMqbtYPBdrkh0bQxKeAerMCp6lzMyE-/view?usp=sharing">this link</a>
##### 2- Put the Model in the `Flask_Deployment` Folder
##### 3- From CMD `flask run`
-------------------------------------------------------------------------------------------
<h1 style="color: blue"><b> Deployment </b></h1>
<img src="images/1.jpg" alt="Simply Easy Learning" >
-------------------------------------------------------------------------------------------
<img src="images/2.jpg" alt="Simply Easy Learning" >
