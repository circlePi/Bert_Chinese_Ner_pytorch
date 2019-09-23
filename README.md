# Bert_Chinese_NER_By_pytorch
A implements of Named Recognition Entity based on BERT in the framework of Pytorch 
## Data
- 2014 people's daliy newpaper
- The preprocessed data is free to download at https://pan.baidu.com/s/17sa7a-u-cDXjbW4Rok2Ntg
## Tips
- Given the BertTokenizer use a greedy longest-match-first algorithm to perform tokenization using a given vocabulary, a word is likely to be splitted into more than one spieces. For example, the input "unaffable" is splitted into ["un", "##aff", "##able"]. This means the number of words processed by BertTokenizer is generally larger than that of the raw inputs. Jocob keeps the first sub_word as the feature sent to crf in his paper, we do so. In fact, in Chinese NER, this case is few. But for robustness, we use a "output_mask" (see preprocessing.data_processor.convert_examples_to_features) to filter the non-first sub_word.
- Note that if your raw inputs have a word as: "谢ing", it would be tokenized as "谢 ing", instead of "谢 ##ing", which couldn't be filtered by the "output_mask". So we need perform another preprocessing on our raw data to avoid this.
## result
classify_report:
precision recall f1-score support

      0       0.98      0.97      0.98     43652
      1       0.98      0.98      0.98     84099
      2       1.00      1.00      1.00     43323
      3       1.00      1.00      1.00    117899
      4       0.96      0.88      0.92      3569
      5       0.86      0.99      0.92      7847
      6       0.99      0.99      0.99     49667
      7       0.98      0.99      0.99     78875
      8       1.00      1.00      1.00   4055215
      avg / total 1.00 1.00 1.00 4484146
Epoch 2 - train_loss: 0.047266 - eval_loss: 0.037106 - train_acc:0.997468 - eval_acc:0.997581 - eval_f1:0.973249
