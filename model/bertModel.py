import warnings
from torch import nn
from model.self_attention import SelfAttention
import torch

warnings.filterwarnings("ignore")

from transformers import (
    BertModel,
    BertConfig,
    DNATokenizer, BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

class BCDB(nn.Module):
    def __init__(self, model_path):
        super(BCDB, self).__init__()
        self.config = BertConfig.from_pretrained(model_path, finetuning_task="dnaprom")
        self.bert = BertModel.from_pretrained(model_path, config=self.config)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=128, kernel_size=5, padding=2, stride=1),
            nn.ELU(),
            nn.BatchNorm1d(num_features=128)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2, stride=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_features=128)
        )
        self.dropout1 = nn.Dropout(p=0.3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=5, padding=2, stride=1)

        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True, bidirectional=True)

        # self.attention1 = SelfAttention(100, 768, 8)
        self.attention = SelfAttention(100, 128, 8)

        self.output = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=(1,)),
            nn.Flatten(start_dim=1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_data = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_data = bert_data[0]
        lstm1, (hd, cn) = self.lstm(bert_data)
        lstm1 = lstm1[:, :, :768] + lstm1[:, :, 768:]

        # attn_seq_1 = self.attention1(bert_data)

        conv1_seq = self.conv1(lstm1.transpose(1, 2))
        maxpool1 = self.maxpool1(conv1_seq)
        maxpool1 = self.dropout1(maxpool1)

        attn_seq, attn = self.attention(maxpool1)
        output1 = self.output(attn_seq)

        conv2_seq = self.conv2(maxpool1)
        output2 = self.output(conv2_seq)

        return output1, output2, attn

