# BCDB: A Dual-Branch Network Based on Transformer for Predicting Transcription Factor Binding Sites

First, you need to download DNABERT. You can download it from https://github.com/jerryji1993/DNABERT. This project uses DNABERT6, which employs kmer=6. After downloading, place it in the current directory. Then, enter the following command to process the original TF data into K-mer format.
```
python process_690.py --kmer=6 --file_path=tf_data --output_path=./process_data/6
```
Where file_path is the path to the original TF data, and output_path is the path for the preprocessed data. Finally, run the following.
```
train.py
```
You need to change the path to the DNABERT in the train.py file to your download path.
