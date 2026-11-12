import os
import random
from typing import List
import fasttext
import numpy as np
from helper.time_util import cost_time
from tqdm.auto import tqdm
"""
    adapted from:
    
    Zhang S, Jin P, Lin Z, et al. Robust Failure Diagnosis of Microservice System through Multimodal Data[J]. 
    https://arxiv.org/abs/2302.10512
"""

class FastTextEncoder:
    def __init__(
        self,
        modality,
        nodes,
        types,
        embedding_dim=100,
        epochs=5,
        random_state=1,
        logger=None,
    ):
        self.dim = embedding_dim
        self.epochs = epochs
        self.random_state=random_state
        self.logger = logger

        self.nodes = nodes
        self.types = types
        self.modality = modality

    def build_datasets(self, data_set, labels, model):
        final_data = data_set.copy()

        group_bar = tqdm(
            total=len(self.types) * len(self.nodes),
            desc=f"{self.modality}: augment corpus",
            unit="group",
            dynamic_ncols=True,
            ascii=True,
            leave=False,
        )
        for event_type in self.types:
            for node in self.nodes:
                expected_label = f'__label__{node}::{self.types.index(event_type)}'
                idxs = [i for i, label in enumerate(labels)
                    if label == expected_label]
                sample_count = len(idxs)
                if sample_count == 0:
                    group_bar.update(1)
                    continue
                anomaly_texts = [data_set[idx] for idx in idxs]
                loop = 0
                sample_num = 1000
                while sample_count < sample_num:
                    loop += 1
                    if loop >= 10 * sample_num:
                        break
                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                    chosen_text_splits = chosen_text.split()
                    if len(chosen_text_splits) < 1:
                        continue
                    edit_event_ids = random.sample(range(len(chosen_text_splits)), 1)
                    for event_id in edit_event_ids:
                        neighbours = model.get_nearest_neighbors(chosen_text_splits[event_id])
                        if neighbours:
                            chosen_text_splits[event_id] = neighbours[0][-1]
                    final_data.append(
                        ' '.join(
                            chosen_text_splits) + f'\t{expected_label}\n')
                    sample_count += 1
                group_bar.set_postfix(samples=len(final_data))
                group_bar.update(1)
        group_bar.close()

        return final_data


    def save_to_txt(self, data_set, filename):
        os.makedirs('./tmp', exist_ok=True)
        path = f'./tmp/{filename}'
        with open(path, 'w') as f:
            for text in data_set:
                f.write(text) 

        return path

    @cost_time
    def fit(self, data_set: dict, labels: dict):
        if self.logger:
            self.logger.info("[%s FastText 1/4] Prepare base corpus", self.modality)
        # Keep an explicit training token so a modality containing empty
        # documents can still train. Empty inference documents remain zero.
        data_set = [' '.join(events) or '__empty__' for events in data_set]
        data_set = [f'{text}\t{labels[idx]}\n' for idx, text in enumerate(data_set)]
        train_pth = self.save_to_txt(data_set, f'{self.modality}-train.txt')
        if self.logger:
            self.logger.info(
                "[%s FastText 2/4] Train base model: documents=%d, epochs=%d",
                self.modality,
                len(data_set),
                self.epochs,
            )
        model = fasttext.train_supervised(
            train_pth,
            dim=self.dim,
            minCount=1, 
            minn=0, maxn=0, epoch=self.epochs
        )
        # model = fasttext.train_unsupervised(
        #     train_pth,
        #     dim=self.dim,
        #     minCount=1, 
        #     minn=0, maxn=0, epoch=self.epochs
        # )

        # aug
        if self.logger:
            self.logger.info("[%s FastText 3/4] Build augmented corpus", self.modality)
        aug_data_set = self.build_datasets(data_set, labels, model)
        aug_train_pth = self.save_to_txt(aug_data_set, f'{self.modality}-train-aug.txt')
        if self.logger:
            self.logger.info(
                "[%s FastText 4/4] Retrain with %d augmented documents",
                self.modality,
                len(aug_data_set),
            )
        model = fasttext.train_supervised(
            aug_train_pth,
            dim=self.dim,
            minCount=1, 
            minn=0, maxn=0, epoch=self.epochs
        )
        # model = fasttext.train_unsupervised(
        #     aug_train_pth,
        #     dim=self.dim,
        #     minCount=1, 
        #     minn=0, maxn=0, epoch=self.epochs
        # )

        # event embedding
        self.event_dic = {}
        for e in tqdm(
            model.words,
            desc=f"{self.modality}: extract vectors",
            unit="word",
            dynamic_ncols=True,
            ascii=True,
            leave=False,
        ):
            self.event_dic[e] = model[e]
        if self.logger:
            self.logger.info(
                "%s FastText ready: vocabulary=%d",
                self.modality,
                len(self.event_dic),
            )
        

    def get_sentence_embedding(self, text: List[str]) -> List[float]:
        text = ' '.join(text)
        
        # senetence embedding
        sen_emb = np.zeros(self.dim, dtype='float32')
        if text != '':
            words = list(set(text.split(' ')))
            for word in words:
                if word in self.event_dic:
                    sen_emb = sen_emb + np.array(self.event_dic[word])

        return sen_emb
