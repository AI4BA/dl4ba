# dl4ba
Our experiment is conducted according to the following technical roadmap.
<img src="[https://github.com/AI4BA/roadmap.pdf](https://github.com/AI4BA/dl4ba/blob/main/roadmap.pdf)" alt="roadmap"/>
## Getting Started
### Requirements
- nltk==3.8.1
- numpy==1.23.5
- pandas==1.5.3
- PyYAML==6.0
- scikit-learn==1.2.2
- torch==1.12.1
- transformers==4.20.1
- yaml==0.2.5
- allennlp==2.10.1
- matplotlib==3.7.1
- json==0.9.6

### Pre-Trained Models
- Word2Vec is GoogleNews-vectors-negative300.bin.
- GloVe is glove.840B.300d.txt.
- NextBug could be downloaded in https://github.com/xiaotingdu/DeepSIM.
- BERT could be downloaded in https://github.com/google-research/BERT.
- ELMo could be downloaded in https://s3-us-west-2.amazonaws.com/allennlp/models/elmo.

### Directory Structure
- data analysis: It contains code used to analyse results, such as code for Cliff's Delta, code for Wilcoxon signed-rank test, code for drawing plots, and so on.
- datasets: It consist of three widely-used datasets in bug assignment.
- res: It is a set of results about all research questions.
- models.py: Code that implements TextCNN, LSTM, Bi-LSTM, LSTM with attention, Bi-LSTM with attention, and MLP.
- parameters.json: Parameters for training models
- train.py: Code for training models
- utils.py: Code for text preprocessing, word vectors generating, and so on.
