# TODO: complete this file.
from matrix_factorization import *
from item_response import *
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

# new function
def average_predictions(data, *matrices):
    predictions = np.zeros(len(data["user_id"]))
    for matrix in matrices:
        user_ids = data["user_id"]
        question_ids = data["question_id"]
        predictions += np.array([matrix[user_id, question_id] for user_id, question_id in zip(user_ids, question_ids)])
    avg_predictions = predictions / len(matrices)
    return avg_predictions

def get_accuracy(data, prediction):
    total = 0
    correct = 0
    for i in range(len(data['is_correct'])):
        total += 1
        pred = prediction[i]
        if round(pred) == data["is_correct"][i]:
            correct += 1
    return correct / total

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

    # average after calculate accuracy
    # acc_val_a = sparse_matrix_evaluate(val_data, matrix_a)
    # acc_val_b = sparse_matrix_evaluate(val_data, matrix_b)
    # acc_val_c = sparse_matrix_evaluate(val_data, matrix_c)
    # avg_val = (acc_val_a + acc_val_b + acc_val_c) / 3
    # print(f"avg acc for validation is: {avg_val}")
    #
    # acc_test_a = sparse_matrix_evaluate(test_data, matrix_a)
    # acc_test_b = sparse_matrix_evaluate(test_data, matrix_b)
    # acc_test_c = sparse_matrix_evaluate(test_data, matrix_c)
    # avg_test = (acc_test_a + acc_test_b + acc_test_c) / 3
    # print(f"avg acc for test is: {avg_test}")



    # Version 1
    avg_val_predictions = average_predictions(val_data, matrix_a, matrix_b, matrix_c)
    avg_test_predictions = average_predictions(test_data, matrix_a, matrix_b, matrix_c)

    # avg_val_acc = np.mean(avg_val_predictions)
    # avg_test_acc = np.mean(avg_test_predictions)

    avg_val_acc = get_accuracy(val_data, avg_val_predictions)
    avg_test_acc = get_accuracy(test_data, avg_test_predictions)

    print(f"Average validation accuracy: {avg_val_acc}")
    print(f"Average test accuracy: {avg_test_acc}")

    # Version 2
    combined_matrix = (matrix_a + matrix_b + matrix_c) / 3
    acc_v = sparse_matrix_evaluate(val_data, combined_matrix)
    acc_t = sparse_matrix_evaluate(train_data, combined_matrix)
    print(f"V2 validation acc:{acc_v}, test acc:{acc_t}")






if __name__ == "__main__":
    main()
