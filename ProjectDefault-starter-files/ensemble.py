# TODO: complete this file.
from matrix_factorization import *
from knn import *
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def matrix_factorization(data):
    reconstruct_matrix = als(data, 25, 0.05, 150000)
    return reconstruct_matrix

def boostrapping(data):
    n = len(data["user_id"])
    index = np.random.randint(n, size=len(data['user_id']))
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    generated_data = {
        'user_id': [user_id[i] for i in index],
        'question_id': [question_id[i] for i in index],
        'is_correct': [is_correct[i] for i in index]
    }
    return generated_data

def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    data_a = boostrapping(train_data)
    data_b = boostrapping(train_data)
    data_c = boostrapping(train_data)

    matrix_a = matrix_factorization(data_a)
    matrix_b = matrix_factorization(data_b)
    matrix_c = matrix_factorization(data_c)


    combined_matrix = (matrix_a + matrix_b + matrix_c) / 3
    acc_v = sparse_matrix_evaluate(val_data, combined_matrix)
    acc_t = sparse_matrix_evaluate(train_data, combined_matrix)
    print(f"validation acc:{acc_v}, test acc:{acc_t}")






if __name__ == "__main__":
    main()
