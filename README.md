# Bert_Chinese_NER_By_pytorch
A implements of Named Recognition Entity based on BERT in the framework of Pytorch 
## Data
- 2014 people's daliy newpaper
- The preprocessed data is free to download at https://pan.baidu.com/s/17sa7a-u-cDXjbW4Rok2Ntg
## Tips
- Given the BertTokenizer use a greedy longest-match-first algorithm to perform tokenization using a given vocabulary, a word is likely to be splitted into more than one spieces. For example, the input "unaffable" is splitted into ["un", "##aff", "##able"]. This means the number of words processed by BertTokenizer is generally larger than that of the raw inputs. Jocob keeps the first sub_word as the feature sent to crf in his paper, we also do so. In fact, in Chinese NER, this case is few. But for robustness, we use a "output_mask" (see preprocessing.data_processor.convert_examples_to_features) to filter the non-first sub_word.
- Note that if your raw inputs have a word as: "谢ing", it would be tokenize as "谢 ing", instead of "谢 ##ing", which couldn't be filtered by the "output_mask". So we need perform another preprocessing on our raw data to avoid this.
