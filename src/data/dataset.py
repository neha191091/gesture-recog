from src import config
import numpy as np
import glob


class Dataset:
    key_template = '{:d}_{:d}_{:d}_{:d}'

    def __init__(self, split, batch_size=20, num_features=3):
        train, val, test = split
        user_idxs = np.random.permutation(np.arange(0, config.Num.users.value))

        self.train_idxs = user_idxs[:int(config.Num.users.value * train)]
        self.val_idxs = user_idxs[int(config.Num.users.value * train):int(config.Num.users.value * train)+int(config.Num.users.value * val)]
        self.test_idxs = user_idxs[int(config.Num.users.value * train)+int(config.Num.users.value * val):]

        self.largest_seq_len = 0
        self.smallest_seq_len = 0

        self.min_value = np.ones([num_features])*100
        self.max_value = -1*np.ones([num_features])*100

        self.data_train = self._return_data_dictionary(config.SplitType.train.value)
        self.data_val = self._return_data_dictionary(config.SplitType.val.value)
        self.data_test = self._return_data_dictionary(config.SplitType.test.value)

        self.train_keys = list(self.data_train.keys())
        self.val_keys = list(self.data_val.keys())
        self.test_keys = list(self.data_test.keys())

        self.batch_idx = 0
        self.batch_size = batch_size

        self.num_features = num_features

        self._shuffle_keys()

    def get_data_point_velocity(self, data_point):
        vel = [np.sum(data_point[:k], axis=0) for k in range(len(data_point))]
        vel = np.array(vel)
        return vel

    def get_data_point_pos(self, data_point):
        vel = self.get_data_point_velocity(data_point)
        pos = [np.sum(vel[:k], axis=0) for k in range(len(vel))]
        pos = np.array(pos)
        return pos

    def get_random_data_point(self, split_type):
        if split_type == config.SplitType.train.value:
            return np.random.choice(list(self.data_train.values()))
        elif split_type == config.SplitType.val.value:
            return np.random.choice(list(self.data_val.values()))
        else:
            return np.random.choice(list(self.data_test.values()))

    def get_random_data_point_for_class(self, split_type, num_class):
        d = np.random.randint(0, config.Num.days.value)
        c = num_class
        r = np.random.randint(0, config.Num.repeat.value)
        if split_type == config.SplitType.train.value:
            u = np.random.choice(self.train_idxs)
            data = self.data_train
        elif split_type == config.SplitType.val.value:
            u = np.random.choice(self.val_idxs)
            data = self.data_val
        else:
            u = np.random.choice(self.test_idxs)
            data = self.data_test

        return data[Dataset.key_template.format(u, d, c, r)]

    def get_padded_data_points_flat(self, split_type, normalize=True):
        if split_type == config.SplitType.train.value:
            data_dic = self.data_train
        elif split_type == config.SplitType.val.value:
            data_dic = self.data_val
        else:
            data_dic = self.data_test
        data = []
        cls = []
        seq_size = []
        for k, v in data_dic.items():
            data.append(np.pad(v, ((0, self.largest_seq_len-v.shape[0]), (0, 0)), 'constant'))
            cls.append(int(k[4:][0]))
            seq_size.append(v.shape[0])
        data = np.array(data)

        # Normalize
        if normalize:
            data = self._normalize(data)

        data = data.reshape([data.shape[0], -1])
        return np.array(data), np.array(cls), np.array(seq_size)

    def get_padded_batch_flat(self, split_type, normalize=True):
        if split_type == config.SplitType.train.value:
            vals = self.data_train
            keylist = self.train_keys
        elif split_type == config.SplitType.val.value:
            vals = self.data_val
            keylist = self.val_keys
        else:
            vals = self.data_test
            keylist = self.test_keys

        keys = keylist[self.batch_idx:self.batch_idx + self.batch_size]
        self.batch_idx = self.batch_idx+self.batch_size
        if self.batch_idx > len(keylist) - self.batch_size:
            self.batch_idx = 0
            self._shuffle_keys()

        data = []
        cls = []
        seq_size = []
        for k in keys:
            data.append(np.pad(vals[k], ((0, self.largest_seq_len-vals[k].shape[0]), (0, 0)), 'constant'))
            cls.append(int(k[4:][0]))
            seq_size.append(vals[k].shape[0])
        data = np.array(data)

        # Normalize
        if normalize:
            data = self._normalize(data)

        data = data.reshape([data.shape[0], -1])

        return data, np.array(cls), np.array(seq_size)

    def _return_data_dictionary(self, split_type):
        largest_seq_len = 0
        smallest_seq_len = 0

        min_value = self.min_value
        max_value = self.max_value

        data_points = {}
        if split_type == config.SplitType.train.value:
            user_idxs = self.train_idxs
        elif split_type == config.SplitType.val.value:
            user_idxs = self.val_idxs
        else:
            user_idxs = self.test_idxs
        for user in user_idxs:
            for day in range(config.Num.days.value):
                firstfile = config.FileTemplate.format(user + 1, day + 1, '*', 1, 1)
                firstfile = glob.glob(firstfile)[0]
                letter = firstfile[len(config.DirPrefixTemplate.format(1, 1))]
                for gesture in range(config.Num.classes.value):
                    for repeat in range(config.Num.repeat.value):
                        file = config.FileTemplate.format(user + 1, day + 1, letter, gesture + 1, repeat + 1)
                        fp = open(file)
                        lines = fp.read().split("\n")
                        data_point = [seq.split(' ') for seq in lines[:-1]]
                        data_point = np.array(data_point, dtype=np.float)

                        data_points[str(user) + '_' + str(day) + '_' + str(gesture) + '_' + str(repeat)] = data_point

                        min_value = (data_point.min(axis=0)<min_value)*data_point.min(axis=0) + (data_point.min(axis=0)>min_value)*min_value

                        max_value = (data_point.max(axis=0)>max_value)*data_point.max(axis=0) + (data_point.max(axis=0)<max_value)*max_value

                        if largest_seq_len < data_point.shape[0]:
                            largest_seq_len = data_point.shape[0]
                        if smallest_seq_len > data_point.shape[0]:
                            smallest_seq_len = data_point.shape[0]

        if self.largest_seq_len < largest_seq_len:
            self.largest_seq_len = largest_seq_len
        if self.smallest_seq_len > smallest_seq_len:
            self.smallest_seq_len = smallest_seq_len

        self.min_value = (self.min_value < min_value) * self.min_value + (self.min_value > min_value) * min_value
        self.max_value = (self.max_value > max_value) * self.max_value + (self.max_value < max_value) * max_value

        return data_points

    def _normalize(self, data):

        size, seq_len, num_features = data.shape

        data = data.reshape([-1, num_features])
        data = data - self.min_value/(self.max_value - self.min_value)

        return data.reshape([size, seq_len, num_features])

    def _shuffle_keys(self):
        self.train_keys = np.random.permutation(self.train_keys)
        self.val_keys = np.random.permutation(self.val_keys)
        self.test_keys = np.random.permutation(self.test_keys)



