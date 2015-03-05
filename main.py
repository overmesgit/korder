import random

test_data = [
    [0, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 0],
]

factors = 5
V = [
    [random.random() for i in range(0, factors)] for j in range(len(test_data[0]))
]


def row_multiply(a, b):
    return sum(map(lambda d: d[0]*d[1], zip(a, b)))


def get_f(user_index, current_item):
    user_scores = test_data[user_index]
    Vd = V[current_item]
    # each positive user item features * current_item features
    result_sum = sum(row_multiply(Vd, V[r]) for r in range(len(user_scores)) if user_scores[r] > 0)
    return result_sum/sum(1 for r in user_scores if r > 0)


def get_positive_item_d():
    random_user_index = random.randint(0, len(test_data) - 1)
    user_scores = test_data[random_user_index]
    # Can be random subset of scores or only one score
    Fdu = [(i, get_f(random_user_index, i)) for i in range(len(user_scores)) if user_scores[i] > 0]
    Fdu.sort(reverse=True, key=lambda x: x[1])

    # Can be probability distribution P
    random_position = random.randint(0, len(Fdu) - 1)
    return random_user_index, Fdu[random_position][0]


def get_n(user_index, current_item):
    N = 1
    user_scores = test_data[user_index]
    # D - Du, all scores - user scores
    not_Du_indexes = [i for i in range(len(user_scores)) if not user_scores[i]]
    not_scores_item = random.choice(not_Du_indexes)
    while get_f(user_index, not_scores_item) < get_f(user_index, current_item) - 1 and N <= len(not_Du_indexes):
        N += 1
        not_scores_item = random.choice(not_Du_indexes)
    return not_scores_item, N


def weight_F(n):
    return sum(1/i for i in range(1, int(n)))


def error_func(user_index, item_index, not_scored_item, N):
    user_scores = test_data[user_index]
    not_Du_len = sum(1 for s in user_scores if not s)
    return weight_F(not_Du_len/N)*max(0, 1 + get_f(user_index, not_scored_item) - get_f(user_index, item_index))


def enforce_constraints(item):
    current_V = V[item]
    norm = sum(current_V)/len(current_V)
    C = 0.6
    if norm > C:
        V[item] = [C*v/norm for v in current_V]


last_errors_sum = 0
for e in range(500):
    current_user, scored_item = get_positive_item_d()
    not_scored_item, N = get_n(current_user, scored_item)

    if get_f(current_user, not_scored_item) > get_f(current_user, scored_item) - 1:
        error = error_func(current_user, scored_item, not_scored_item, N)
        last_errors_sum += error

        V[not_scored_item] = [v - 0.01*error for v in V[not_scored_item]]
        V[scored_item] = [v + 0.01*error for v in V[scored_item]]
        enforce_constraints(not_scored_item)
        enforce_constraints(scored_item)

    if e % 10 == 0:
        print(last_errors_sum/10)
        last_errors_sum = 0

for i, user in enumerate(test_data):
    print()
    for j, item in enumerate(user):
        print('{:3.2f}'.format(get_f(i, j)), end=' ')