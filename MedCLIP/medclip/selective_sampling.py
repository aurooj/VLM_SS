import json
import os
import random
import math
from tqdm.auto import tqdm



class SelectiveSampling(object):
    def __init__(self, data):   
        '''
        Parameters: 
        data: list of dictionaries
            Each dictionary in this list belongs to a data instance, and
            can have an arbitrary number of columns but expects the key 'group' in each item's dictionary.
            Example:
                Format: [
                { 'colA':<value>,
                'colB':<value>,
                'group':['a', 'b']
                }, 
                { 'colA':<value>,
                'colB':<value>,
                'group':['a', 'b']
                }, 
                ...
                ]
        '''     
        self.ann = data
        
        print("created SelectiveSampling object.")

    
    def __len__(self):
        return len(self.ann)


    def get_sort_by_groups_dict(self):
    #expecting a list of items dictionaries
        data = self.ann
        if data is not None:
            new_data_grouped = {}
            for item in data:
                group_key = "_".join(item['group'])
                if group_key not in new_data_grouped:
                    new_data_grouped[group_key] = [item]
                else:
                    new_data_grouped[group_key].append(item)
            return new_data_grouped
        else:
            raise Exception("Error, invalid data, data=None")
            return {}

    def get_sorted_counter(self, list_, reverse=True):
        '''
        Parameters
        -------------
        list_: List of elements (groups for all data instances for selective sampling case)
                elem in list_ is a list of strings, e.g., ['A', 'B', 'C'] 
                For mammography reports, elem will be a list of extracted image descriptors from the text report (group),
                e.g.,  ['scattered fibroglandular densities', 'benign calcification']
            
        reverse: bool, default=True
                The default sorting order is descending. Set reverse=False for ascending order sorting
        
        Returns
        ------------
        sorted_counter: A sorted counter of unique groups based on their frequency
        '''
    
        counter_ = {}
        # print(list_)
        for elem in list_:
            # print("Elem:")
            # print(elem)
            elem = "_".join(elem)

            if elem in counter_:
                counter_[elem]+=1
            else:
                counter_[elem]=1
        sorted_counter = sorted(counter_.items(), key=lambda x:x[1], reverse=reverse)
        return sorted_counter

    def get_groups_list(self, data):
        '''
        Parameters
        -------------
        data: List of data samples dictionaries 
            This list is expected to have the key 'group' for each data instance in this list
        
        
        Returns
        ------------
        groups: List of groups for data
        '''
        groups = []
        if data is not None:
            for dt in data:
                groups.append(dt['group'])

        return groups

    def shuffle(self, bs=8, rare_grp_ratio=0.375, batch_shuffle=False, num_frequent_grps=20):
        '''
        todo: sample randomly from data for each sampled group
        to do so, we need to store data grouped based on groups
        add to shuffled_data till we reach the final data length
        Note: this approach does not guarantee to cover all samples in each epoch 
        but it makes sure to shuffle the data randomly in a way that same group is not repeated in a batch, 
        overall, we hope that after many epochs, the model would have seen all the data. 

        Parameters
        -------------
        bs: int, default=8
            batch size
        
        rare_grp_ratio: float, default=0.375
            Percentage of samples from rare groups we want to keep in the mini-batch
            For a batch size of 8, we keep 3 samples from rare groups in a mini-batch, i.e.,
            3/8 = 0.375

        batch_shuffle: bool, default=False
            Once the mini-batch is sampled using selective sampling, shuffle the samples within the mini batch
            Default behavior is to set to False, our experiments show better retrieval performance of models with the default value.

        num_frequent_grps: int, default=20
            Value of num_frequent_grps depends on the training data distribution and is set accordingly. 
            For training ALBEF and MEDCLIP on our dataset, we set its value to 20 based on the empirical selection of number of groups 
            that we consider as frequent groups.

        
        Returns
        ------------
        shuffled_data2: List of shuffled data using selective sampling for mini-batches

        '''
        print("shuffling using selective sampling..")

        boundary = math.ceil(bs * rare_grp_ratio) #boundary tells how many samples we want from less frequent groups; 
        #we sample bs-boundary data samples from frequent groups

        data = self.ann

        self.all_groups = self.get_groups_list(data=data)
        grp_counter = self.get_sorted_counter(self.all_groups)

        #get frequent groups
        frequent_grps = [gc[0] for gc in grp_counter[:num_frequent_grps]]

        remaining_grps = [gc[0] for gc in grp_counter[num_frequent_grps:]]

        data_grouped = self.get_sort_by_groups_dict()
        groups = list(data_grouped.keys())

        shuffled_data = []
        if groups is not None and data is not None:
            for l in tqdm(range(int(len(data)/bs))):
                sampled_grps_f = random.sample(frequent_grps, k=bs-boundary)
                sampled_grps_r = random.sample(remaining_grps, k=boundary)

                sampled_grps = sampled_grps_f + sampled_grps_r
            
                batch = [random.sample(data_grouped[grp], k=1) for grp in sampled_grps]

                if batch_shuffle:
                    random.shuffle(batch)

                shuffled_data.extend(batch)
        shuffled_data2 = [datum[0] for datum in shuffled_data]
        self.ann = shuffled_data2
        return shuffled_data2

    def shuffle_all(self, bs=8):

        '''
        shuffle_all randomly samples groups to pick example from for a batch
        and does not care about group frequency: rare vs frequent
        to do so, we need to store data grouped based on groups
        add to shuffled_data till we reach the final data length
        Note: this approach does not guarantee to cover all samples in each epoch 
        but it makes sure to shuffle the data randomly in a way that same group is not repeated in a batch, 
        overall, we hope that after many epochs, the model would have seen all the data. 

        Parameters
        -------------
        bs: int, default=8
            batch size
        
        
        Returns
        ------------
        shuffled_data2: List of shuffled data using selective sampling for mini-batches

        '''

        data = self.ann

        self.all_groups = self.get_groups_list(data=data)
        grp_counter = self.get_sorted_counter(self.all_groups)

        unique_grps = [gc[0] for gc in grp_counter]
        
        data_grouped = self.get_sort_by_groups_dict()
        groups = list(data_grouped.keys())

        shuffled_data = []
        if groups is not None and data is not None:
            for l in tqdm(range(int(len(data)/bs))):
                sampled_grps = random.sample(unique_grps, k=bs)
            
                batch = [random.sample(data_grouped[grp], k=1) for grp in sampled_grps]

                shuffled_data.extend(batch)
        print(len(shuffled_data))
        shuffled_data2 = [datum[0] for datum in shuffled_data]

        self.ann = shuffled_data2
        return shuffled_data2
