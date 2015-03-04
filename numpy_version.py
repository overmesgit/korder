from functools import lru_cache
import random
import numpy as np
import scipy as sp
from scipy.sparse import linalg
from scipy import sparse

test_data = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1],
]

test_data_array = np.array(test_data)
test_data_sparse = sparse.csr_matrix(test_data_array)

factors = 20
users_count = test_data_sparse.shape[0]
items_count = test_data_sparse.shape[1]
items_indexes = set(range(items_count))
V_array = np.random.randint(1, 50, (items_count, factors))/100

def row_multiply(a, b):
    return sum(map(lambda d: d[0]*d[1], zip(a, b)))


def get_f(user_scores_indexes, current_item):
    Vd = V_array[current_item]
    # each positive user item features * current_item features
    result_sum = (V_array[user_scores_indexes]*Vd).sum()
    return result_sum/user_scores_indexes.size


def get_positive_item_d():
    random_user_index = random.randint(0, users_count - 1)
    user_scores_array = test_data_sparse[random_user_index]
    # Can be random subset of scores or only one score
    Fdu = []
    for item_index in user_scores_array.indices:
        Fdu.append((item_index, get_f(user_scores_array.indices, item_index)))
    Fdu.sort(reverse=True, key=lambda x: x[1])

    # Can be probability distribution P
    random_position = random.randint(0, len(Fdu) - 1)
    return user_scores_array, Fdu[random_position][0]


def get_n(user_indexes, scored_item):
    N = 1
    # D - Du, all scores - user scores
    not_Du_indexes = items_indexes - set(user_indexes)
    not_scored_item = random.sample(not_Du_indexes, 1)[0]
    while get_f(user_indexes, not_scored_item) < get_f(user_indexes, scored_item) - 1 and N <= len(not_Du_indexes):
        N += 1
        not_scored_item = random.sample(not_Du_indexes, 1)[0]
    return not_scored_item, N


@lru_cache(maxsize=128)
def weight_F(n):
    return sum(1/i for i in range(1, n))


def error_func(user_indexes, scored_item, not_scored_item, N):
    not_Du_len = len(items_indexes - set(user_indexes))
    return weight_F(int(not_Du_len/N))*max(0, 1 + get_f(user_indexes, not_scored_item) - get_f(user_indexes, scored_item))


C = 0.4
def enforce_constraints(item):
    current_V = V_array[item]
    norm = current_V.sum()/current_V.size
    if norm > C:
        V_array[item] = C*V_array[item]/norm


last_errors_sum = 0
for e in range(5000):
    user_scores_array, scored_item = get_positive_item_d()
    user_indexes = user_scores_array.indices
    not_scored_item, N = get_n(user_indexes, scored_item)
    if get_f(user_indexes, not_scored_item) > get_f(user_indexes, scored_item) - 1:
        error = error_func(user_indexes, scored_item, not_scored_item, N)
        last_errors_sum += error

        V_array[not_scored_item] = V_array[not_scored_item] - 0.01*error*V_array[not_scored_item]
        V_array[scored_item] = V_array[scored_item] + 0.01*error*V_array[scored_item]
        enforce_constraints(not_scored_item)
        enforce_constraints(scored_item)

    if e % 100 == 0:
        print(last_errors_sum/100)
        last_errors_sum = 0

for u in range(users_count):
    print()
    for i in range(items_count):
        print('{:3.2f}'.format(get_f(test_data_sparse[u].indices, i)), end=' ')