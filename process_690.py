import argparse
import csv
import os
import numpy as np
import random
import pandas as pd


def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string) - kmer:
        sentence += original_string[i:i + kmer] + " "
        i += stride

    return sentence[:-1].strip("\"")


def Process(args):
    path = args.file_path
    all_folders = os.listdir(path)

    count = 0

    for folder in all_folders:
        # load data
        train_path = os.path.join(args.file_path, folder, "train.data")
        test_path = os.path.join(args.file_path, folder, "test.data")

        train_data = pd.read_csv(train_path, header=None, sep=' ')
        test_data = pd.read_csv(test_path, header=None, sep=' ')

        train_sequences = np.array(train_data.loc[:, 1])
        test_sequences = np.array(test_data.loc[:, 1])
        train_labels = np.array(train_data.loc[:, 2])
        test_labels = np.array(test_data.loc[:, 2])

        train_sequences = train_sequences.reshape(train_sequences.shape[0], 1)
        test_sequences = test_sequences.reshape(test_sequences.shape[0], 1)
        train_labels = train_labels.reshape(train_labels.shape[0], 1)
        test_labels = test_labels.reshape(test_labels.shape[0], 1)

        # concat sequence and labels together
        trains = list(np.concatenate((train_sequences, train_labels), axis=1))
        tests = list(np.concatenate((test_sequences, test_labels), axis=1))


        random.seed(24)
        random.shuffle(trains)
        random.shuffle(trains)
        random.shuffle(tests)
        random.shuffle(tests)

        # make output path
        output_path = os.path.join(args.output_path, folder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # write files
        f_train = open(os.path.join(output_path, "train.tsv"), 'wt',newline='')
        tsv_train = csv.writer(f_train, delimiter='\t')
        tsv_train.writerow(["sequence", "label"])
        for i in range(len(trains)):
            sentence = get_kmer_sentence(trains[i][0], args.kmer)
            tsv_train.writerow([sentence, int(trains[i][1])])

        f_dev = open(os.path.join(output_path, "dev.tsv"), 'wt',newline='')
        tsv_dev = csv.writer(f_dev, delimiter='\t')
        tsv_dev.writerow(["sequence", "label"])
        for i in range(len(tests)):
            sentence = get_kmer_sentence(tests[i][0], args.kmer)
            tsv_dev.writerow([sentence, int(tests[i][1])])

        count += 1
        print("Finish %s folders" % (count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmer",
        default=1,
        type=int,
        help="K-mer",
    )
    parser.add_argument(
        "--file_path",
        default=None,
        type=str,
        help="The path of the file to be processed",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="The path of the processed data",
    )
    args = parser.parse_args()

    Process(args)


if __name__ == "__main__":
    main()
