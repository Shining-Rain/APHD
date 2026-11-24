import os
from functools import cached_property

import scipy.sparse as sp
import numpy as np


class DataLoader:
    """Load interaction splits and social edges into sparse matrices."""

    def __init__(self, path, *, include_validation=False):
        self.path = path
        self.include_validation = include_validation
        self._build_compact_user_mapping()
        self.num_users = len(self.internal_to_raw_user)
        self.num_items = 0
        users_list, items_list = self.load_data('train.txt')
        if include_validation:
            val_users, val_items = self.load_data('val.txt')
            users_list.extend(val_users)
            items_list.extend(val_items)
        test_users_list, test_items_list = self.load_data('test.txt')
        follower_list, followee_list = self.load_social_data('social.txt')
        self.num_items += 1
        self.user_list = set(users_list)
        self.train_length = len(users_list)
        self.test_length = len(test_users_list)
        self.social_length = len(follower_list)
        self.num_nodes = self.num_users + self.num_items
        self.train_data = sp.csr_matrix(
            (np.ones_like(users_list), (users_list, items_list)),
            dtype='float64', shape=(self.num_users, self.num_items)
        )
        self.social_data = sp.csr_matrix(
            (np.ones_like(follower_list), (follower_list, followee_list)),
            dtype='float64', shape=(self.num_users, self.num_users)
        )
        self.bipartite_graph = self.sparse_adjacency_matrix()
        self.test_dict = self.build_test(test_users_list, test_items_list)

    @cached_property
    def social_graph(self):
        return self.sparse_mean_social_matrix()

    @cached_property
    def all_positive(self):
        return self.get_user_pos_items(range(self.num_users))

    def _build_compact_user_mapping(self):
        """Map observed user IDs to contiguous indices without remapping items."""
        raw_users = set()
        for name in ('train.txt', 'val.txt', 'test.txt'):
            file_path = os.path.join(self.path, name)
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r') as f:
                for line in f:
                    arr = line.strip().split()
                    if arr:
                        raw_users.add(int(arr[0]))

        social_path = os.path.join(self.path, 'social.txt')
        with open(social_path, 'r') as f:
            for line in f:
                arr = line.strip().split()
                raw_users.update(int(raw_id) for raw_id in arr)

        if not raw_users:
            raise ValueError('Cannot build a compact mapping for an empty user universe.')
        ordered_raw_users = sorted(raw_users)
        self.user_id_map = {
            raw_user: internal_user
            for internal_user, raw_user in enumerate(ordered_raw_users)
        }
        self.internal_to_raw_user = np.asarray(ordered_raw_users, dtype=np.int64)
        self.raw_num_users = ordered_raw_users[-1] + 1

    def _map_user_id(self, raw_user_id):
        try:
            return self.user_id_map[raw_user_id]
        except KeyError as exc:
            raise ValueError(f'Unknown raw user id: {raw_user_id}') from exc

    def _matrix_cache_path(self, stem):
        suffix = '_compact_users'
        return os.path.join(self.path, f'{stem}{suffix}.npz')

    def _read_rows(self, name):
        with open(os.path.join(self.path, name), 'r') as f:
            for line in f:
                fields = line.split()
                if fields:
                    yield int(fields[0]), [int(value) for value in fields[1:]]

    def load_data(self, name):
        users_list, items_list = [], []
        for raw_user, item_ids in self._read_rows(name):
            user_id = self._map_user_id(raw_user)
            if item_ids:
                self.num_items = max(self.num_items, max(item_ids))
            users_list.extend([user_id] * len(item_ids))
            items_list.extend(item_ids)
        return users_list, items_list

    def load_social_data(self, name):
        followers_list, followees_list = [], []
        for raw_user, raw_neighbors in self._read_rows(name):
            follower_id = self._map_user_id(raw_user)
            followers_list.extend([follower_id] * len(raw_neighbors))
            followees_list.extend(self._map_user_id(user) for user in raw_neighbors)
        return followers_list, followees_list

    def sparse_adjacency_matrix(self):
        """Load or construct the degree-normalized user-item interaction matrix."""
        # Selection and refit use different interactions in the same directory.
        stem = 'pre_R_mat_train_val' if self.include_validation else 'pre_R_mat_train'
        npz_path = self._matrix_cache_path(stem)
        try:
            norm_R = sp.load_npz(npz_path)
            print("\t Adjacency matrix loading completed.")
        except Exception:
            R = self.train_data.todok()
            row_sum = np.array(R.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_R = d_mat.dot(R)
            col_sum = np.array(R.sum(axis=0))
            d_inv = np.power(col_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_R = norm_R.dot(d_mat)
            norm_R = norm_R.tocsr()
            sp.save_npz(npz_path, norm_R)
            print("\t Adjacency matrix constructed.")
        self.bipartite_graph = norm_R
        return self.bipartite_graph


    def sparse_mean_social_matrix(self):
        """Scale each destination column by the inverse out-degree of that user."""
        npz_path = self._matrix_cache_path('pre_S_mat')
        try:
            norm_R = sp.load_npz(npz_path)
            print("\t Adjacency matrix loading completed.")
        except Exception:
            R = self.social_data.todok()
            row_sum = np.array(R.sum(axis=1))
            d_inv = np.power(row_sum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_R = R.dot(d_mat)
            norm_R = norm_R.tocsr()
            sp.save_npz(npz_path, norm_R)
            print("\t Adjacency matrix constructed.")
        self.social_graph = norm_R
        return self.social_graph

    def get_user_pos_items(self, users):
        """Return training items to exclude from each user's candidate ranking."""
        return [self.train_data[user].nonzero()[1] for user in users]


    def build_test(self, test_users_list, test_items_list):
        test_data = {}
        for u, i in zip(test_users_list, test_items_list):
            test_data.setdefault(u, []).append(i)
        return test_data

    def load_eval_dict(self, name):
        """Load held-out targets using the existing user mapping and item range."""
        targets = {}
        for raw_user, item_ids in self._read_rows(name):
            if raw_user not in self.user_id_map:
                raise ValueError(f'{name} contains an unknown user id: {raw_user}.')
            user_id = self._map_user_id(raw_user)
            if user_id >= self.num_users or any(i >= self.num_items for i in item_ids):
                raise ValueError(f'{name} contains an id outside the training/test universe.')
            if item_ids:
                targets.setdefault(user_id, []).extend(item_ids)
        return targets

