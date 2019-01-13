import torch.nn as nn
from net.crf import CRF
import numpy as np
from sklearn.metrics import f1_score, classification_report
import config.args as args
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel


class Bert_CRF(PreTrainedBertModel):
    def __init__(self,
                 config,
                 num_tag):
        super(Bert_CRF, self).__init__(config)
        self.bert = BertModel(config)
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tag)
        self.apply(self.init_bert_weights)

        self.crf = CRF(num_tag)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                label_id=None,
                output_all_encoded_layers=False):
        bert_encode, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=output_all_encoded_layers)
        output = self.classifier(bert_encode)
        return output

    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        predicts = predicts.view(1, -1).squeeze()
        predicts = predicts[predicts != -1]
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true==y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)



