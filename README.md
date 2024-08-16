# BCDB: A Dual-Branch Network Based on Transformer for Predicting Transcription Factor Binding Sites
首先需要下载DNABERT，你可以从https://github.com/jerryji1993/DNABERT下载，本项目使用的是DNABERT6，即采用kmer=6
下载完成后将其放入到当前目录下
然后你需要运行process.690,file_path为tf原始数据，output_path为预处理好的数据
```
python process_690.py --kmer=6 --file_path=tf_data --output_path=./process_data/6

```
最后运行
```
train.py
```
