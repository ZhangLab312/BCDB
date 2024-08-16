import torch
import os
import torch.utils.data as loader
import math
from torch.utils.data import random_split
import time
import numpy as np
import torch
import random
from multiprocessing import Pool
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm, trange
import warnings
from torch import nn
from copy import deepcopy
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from transformers import (
    BertModel,
    BertConfig,
    DNATokenizer, BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

#Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed = 0
set_seed(seed)


class Constructor:

    def __init__(self, model, model_name='bert'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model_name
        self.model_path = "./6-new-12w-0"
        self.data_dir = "process_data/6/wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk"
        self.max_seq_length = 100
        self.model_type = "dna"
        self.n_process = 8
        self.do_predict = ""
        self.num_train_epochs = 5
        self.task_name = "dnaprom"
        self.result_dir = None
        self.tokenizer = DNATokenizer.from_pretrained(self.model_path)
        self.batch_size = 32
        self.loss_fn = nn.BCELoss()
        self.predict_dir = 'result/6'

    def train(self, train_dataloader, ValidateLoader):
        t_total = len(train_dataloader) * self.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                # model.named_parameters()获取模型的所有参数及其名称
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=t_total*0.2, num_training_steps=t_total
        )
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.model_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            try:
                global_step = int(self.model_path.split("-")[-1].split("/")[0])
            except:
                global_step = 0
            epochs_trained = global_step // (len(train_dataloader))
            steps_trained_in_current_epoch = global_step % (len(train_dataloader))
        tr_loss, logging_loss = 0.0, 0.0
        train_iterator = trange(
            epochs_trained, self.num_train_epochs, desc="Epoch")

        best = 1
        path = os.path.abspath(os.curdir)
        for _ in train_iterator:
            self.model.train()
            total_loss = 0
            epoch_iterator = tqdm(train_dataloader)
            for step, batch in enumerate(epoch_iterator):
                epoch_iterator.set_description("Epoch %d" % (_ + 1))
                self.model.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": None}

                labels = batch[3].unsqueeze(1)

                output1, output2, attn = self.model(**inputs)
                loss = self.loss_fn(output1.float(), labels.float()) * 0.5 + self.loss_fn(output2.float(),
                                                                                          labels.float()) * 0.5  # model outputs are always tuple in transformers (see doc)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                global_step += 1
            validate_loss = self.validate(ValidateLoader)
            if validate_loss < best:
                best = validate_loss
                model_name = path + '\\' + self.model_name + '.pth'
                torch.save(self.model.state_dict(), model_name)
        return model_name

    # def evaluate(self, model, tokenizer, prefix="", evaluate=True):
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #
    #     eval_dataset = self.load_and_cache_examples(self.task_name, tokenizer, evaluate=evaluate)
    #
    #     eval_batch_size = 32
    #     # Note that DistributedSampler samples randomly
    #     eval_sampler = SequentialSampler(eval_dataset)
    #     eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, drop_last=True)
    #
    #     eval_loss = 0.0
    #     nb_eval_steps = 0
    #     preds = None
    #     probs = None
    #     out_label_ids = None
    #     loss_fn = nn.BCELoss()
    #     y = []
    #     for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #         model.eval()
    #         batch = tuple(t.to(self.device) for t in batch)
    #
    #         with torch.no_grad():
    #             inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": None}
    #             labels = batch[3].reshape(eval_batch_size, -1)
    #             output1, output2,attn = model(**inputs)
    #             outputs = output1 * 0.5 + output2 * 0.5
    #             tmp_eval_loss = loss_fn(output1.float(), labels.float()) * 0.5 + loss_fn(output1.float(),
    #                                                                                      labels.float()) * 0.5
    #
    #             eval_loss += tmp_eval_loss.mean().item()
    #             y.append(tmp_eval_loss.mean().item())
    #         nb_eval_steps += 1
    #         if preds is None:
    #             preds = outputs.detach().cpu().numpy()
    #             out_label_ids = labels.detach().cpu().numpy()
    #         else:
    #             preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
    #             out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    #     eval_loss = eval_loss / nb_eval_steps
    #     predicted_value = np.array(preds).squeeze(1)
    #     true_label = out_label_ids.squeeze(1)
    #     accuracy, roc_auc, pr_auc = self.calculate(predicted_value, true_label)
    #     return accuracy, roc_auc, pr_auc

    def validate(self, ValidateLoader):
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in ValidateLoader:
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": None}

                labels = batch[3].unsqueeze(1)

                output1, output2, attn = self.model(**inputs)
                loss = self.loss_fn(output1.float(), labels.float()) * 0.5 + self.loss_fn(output2.float(),
                                                                                          labels.float()) * 0.5  # model outputs are always tuple in transformers (see doc)
                valid_loss.append(loss.item())
            valid_loss_avg = torch.mean(torch.Tensor(valid_loss))

        return valid_loss_avg

    def test(self, TestLoader, model_name):
        self.model.load_state_dict(torch.load(model_name))
        self.model.eval()

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        probs = None
        out_label_ids = None
        loss_fn = nn.BCELoss()
        y = []
        for batch in tqdm(TestLoader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": None}
                labels = batch[3].unsqueeze(1)
                output1, output2, attn = self.model(**inputs)
                outputs = output1 * 0.5 + output2 * 0.5
                tmp_eval_loss = loss_fn(output1.float(), labels.float()) * 0.5 + loss_fn(output2.float(),
                                                                                         labels.float()) * 0.5
                eval_loss += tmp_eval_loss.mean().item()
                y.append(tmp_eval_loss.mean().item())
            nb_eval_steps += 1
            if preds is None:
                preds = outputs.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        predicted_value = np.array(preds).squeeze(1)
        true_label = out_label_ids.squeeze(1)
        return predicted_value, true_label

    def calculate(self, predicted_value, true_label):
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=true_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=true_label)

        precision, recall, _ = precision_recall_curve(probas_pred=predicted_value, y_true=true_label)
        pr_auc = auc(recall, precision)
        return accuracy, roc_auc, pr_auc

    def visualize(self, tokenizer, kmer, model_name):
        self.model.load_state_dict(torch.load(model_name))
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        pred_task_names = (self.task_name,)
        pred_outputs_dirs = (self.predict_dir,)
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)

        for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):

            pred_dataset = self.load_and_cache_examples(pred_task, tokenizer, evaluate=True)

            if not os.path.exists(pred_output_dir):
                os.makedirs(pred_output_dir)

            # Note that DistributedSampler samples randomly
            pred_sampler = SequentialSampler(pred_dataset)
            pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=self.batch_size)

            pred_loss = 0.0
            nb_pred_steps = 0
            batch_size = self.batch_size

            attention_scores = np.zeros([len(pred_dataset), 8, self.max_seq_length, self.max_seq_length])

            for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": None}
                    output1, output2, attn = self.model(**inputs)
                    attention = attn

                    attention_scores[index * batch_size:index * batch_size + len(batch[0]), :, :,
                    :] = attention.cpu().numpy()

            scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

            for index, attention_score in enumerate(attention_scores):
                attn_score = []
                for i in range(1, attention_score.shape[-1] - kmer + 2):
                    attn_score.append(float(attention_score[:, 0, i].sum()))

                for i in range(len(attn_score) - 1):
                    if attn_score[i + 1] == 0:
                        attn_score[i] = 0
                        break

                # attn_score[0] = 0
                counts = np.zeros([len(attn_score) + kmer - 1])
                real_scores = np.zeros([len(attn_score) + kmer - 1])
                for i, score in enumerate(attn_score):
                    for j in range(kmer):
                        counts[i + j] += 1.0
                        real_scores[i + j] += score
                real_scores = real_scores / counts
                real_scores = real_scores / np.linalg.norm(real_scores)

                scores[index] = real_scores

        return scores

    def load_and_cache_examples(self, task, tokenizer, evaluate=False):
        processor = processors[task]()
        output_mode = output_modes[task]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            self.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train",
                list(filter(None, self.model_path.split("/"))).pop(),
                str(self.max_seq_length),
                str(task),
            ),
        )

        if self.do_predict:
            cached_features_file = os.path.join(
                self.data_dir,
                "cached_{}_{}_{}".format(
                    "dev" if evaluate else "train",
                    str(self.max_seq_length),
                    str(task),
                ),
            )
        if os.path.exists(cached_features_file):
            features = torch.load(cached_features_file)
        else:
            label_list = processor.get_labels()
            examples = (
                processor.get_dev_examples(self.data_dir) if evaluate else processor.get_train_examples(self.data_dir)
            )

            print("finish loading examples")

            # params for convert_examples_to_features
            max_length = self.max_seq_length
            pad_on_left = False

            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            pad_token_segment_id = 4 if self.model_type in ["xlnet"] else 0

            if self.n_process == 1:
                features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    label_list=label_list,
                    max_length=max_length,
                    output_mode=output_mode,
                    pad_on_left=pad_on_left,  # pad on the left for xlnet
                    pad_token=pad_token,
                    pad_token_segment_id=pad_token_segment_id, )

            else:
                n_proc = int(self.n_process)
                if evaluate:
                    n_proc = max(int(n_proc / 4), 1)
                print("number of processes for converting feature: " + str(n_proc))
                p = Pool(n_proc)
                indexes = [0]
                len_slice = int(len(examples) / n_proc)
                for i in range(1, n_proc + 1):
                    if i != n_proc:
                        indexes.append(len_slice * (i))
                    else:
                        indexes.append(len(examples))

                results = []

                label_map = {label: i for i, label in enumerate(label_list)}

                for i in range(n_proc):
                    results.append(p.apply_async(convert_examples_to_features, args=(
                        examples[indexes[i]:indexes[i + 1]], tokenizer, max_length, None, label_list, output_mode,
                        pad_on_left,
                        pad_token, pad_token_segment_id, True,)))
                    print(str(i + 1) + ' processor started !')

                p.close()
                p.join()

                features = []
                for result in results:
                    features.extend(result.get())

            torch.save(features, cached_features_file)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    def run(self, ratio=0.8):
        Train_Validate_Set = self.load_and_cache_examples(self.task_name, self.tokenizer, evaluate=False)

        """divide Train samples and Validate samples"""
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(seed))
        TrainLoader = loader.DataLoader(dataset=Train_Set, batch_size=self.batch_size,
                                        drop_last=True, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, batch_size=self.batch_size,
                                           drop_last=True, shuffle=True, num_workers=0)

        TestLoader = loader.DataLoader(
            dataset=self.load_and_cache_examples(self.task_name, self.tokenizer, evaluate=True),
            batch_size=self.batch_size, shuffle=False, num_workers=0)

        model_name = self.train(TrainLoader, ValidateLoader)

        predicted_value, true_label = self.test(TestLoader, model_name)

        accuracy, roc_auc, pr_auc = self.calculate(predicted_value, true_label)
        print("accuracy:{:.8f} roc_auc:{:.8f} pr_auc:{:.8f}".format(accuracy, roc_auc, pr_auc))


from model.bertModel import BCDB

if __name__ == '__main__':
    start_time = time.time()

    #./6-new-12w-0 is the path for the downloaded DNABERT
    Train = Constructor(model=BCDB("./6-new-12w-0"))
    Train.run()
