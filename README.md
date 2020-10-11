# Sentimix2020
## Abstract 
In social-media platforms such as Twitter, Facebook, and Reddit, people prefer to use code-mixed
language such as Spanish-English, Hindi-English to express their opinions. In this paper, we
describe different models we used, using the external dataset to train embeddings, ensembling
methods for Sentimix, and OffensEval tasks. The use of pre-trained embeddings usually helps in
multiple tasks such as sentence classification, and machine translation. In this experiment, we have
used our trained code-mixed embeddings and twitter pre-trained embeddings to SemEval tasks.
We evaluate our models on macro F1-score, precision, accuracy, and recall on the datasets. We
intend to show that hyper-parameter tuning and data pre-processing steps help a lot in improving
the scores. In our experiments, we are able to achieve 0.886 F1-Macro on OffenEval Greek
language subtask post-evaluation, whereas the highest is 0.852 during the Evaluation Period.
We stood third in Spanglish competition with our best F1-score of 0.756. Codalab username is
asking28.

### File Structure
This repository contains Clutter three folders Clutter, Sentimix and OffensEval. Sentimix folder contains Jupyter and corresponding Python files of Spanglish and Hinglish. OffensEval Folder contains code files of OffensEval English Task 1,2,3 and Turkish, Arabic, Danish and Greek languages. Please ignore clutter folder there are repeated entries.
### BibTex
@article{singh2020voice,
  title={Voice@ SRIB at SemEval-2020 Task [9, 12]: Sentiment and Offensiveness detection in Social Media},
  author={Singh, Abhishek and Parmar, Surya Pratap Singh},
  journal={arXiv preprint arXiv:2007.10021},
  year={2020}
}

### External libraries
Sklearn, Numpy, Pandas, Tensorflow, Keras, Beautiful Soup <br />
Run below commands before running files (Pip):<br />
!pip install focal-loss <br />
!pip install keras-tcn==2.8.3 <br />
!pip install keras-multi-head <br />
!pip install keras_metrics <br />
!pip install tqdm <br />
!pip install keras-self-attention <br />
### References
https://github.com/SilentFlame/Named-Entity-Recognition/blob/master/Twitterdata/processedTweets.csv <br />
https://arxiv.org/pdf/1805.11869.pdf <br />
http://ceur-ws.org/Vol-2111/paper5.pdf <br />
https://github.com/sahilswami96/SarcasmDetection_CodeMixed/blob/master/Dataset/Sarcasm_tweets.txt <br />
https://github.com/sahilswami96/SarcasmDetection_CodeMixed/blob/master/Classification_system/build_feature_vector.py <br />
https://www.aclweb.org/anthology/C18-1247.pdf how emotional are you <br />
https://arxiv.org/pdf/1905.12516.pdf multiple datasets <br />
