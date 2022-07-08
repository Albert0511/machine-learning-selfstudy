import numpy as np
import os
import util

from linear_model import LinearModel

sample_info = [("../data/ds1_train.csv", "../data/ds1_valid.csv", "../output/p01b_pred_1.txt"),
               ("../data/ds2_train.csv", "../data/ds2_valid.csv", "../output/p01b_pred_2.txt")]


def check_accuracy(logistic_regression, x_dataset, y_dataset):
    y_dataset = np.reshape(y_dataset, newshape=(-1, 1))
    y_dataset_predict = logistic_regression.predict(x_dataset)
    print("Theta is: ", logistic_regression.theta)
    print("The total number of data is: ", x_dataset.shape[0])
    print("The accuracy on training set is: ", np.mean(y_dataset == y_dataset_predict))


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train, y_train)
    check_accuracy(logistic_regression, x_train, y_train)

    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)
    y_test_out = logistic_regression.predict(x_test)
    check_accuracy(logistic_regression, x_test, y_test)

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    np.savetxt(pred_path, X=y_test_out)

    return y_test_out
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        sample_size = x.shape[0]
        sample_dim = x.shape[1]
        y = np.reshape(y, newshape=(-1, 1))

        # *** START CODE HERE ***

        def hypothesis_func(x_matrix, theta_vec):
            return 1 / (1 + np.exp(-np.matmul(x_matrix, theta_vec)))

        def gradient(x_matrix, y_vec, theta_vec):
            diff_vec = y_vec - hypothesis_func(x_matrix, theta_vec)
            return -1 / sample_size * np.matmul(np.transpose(x), diff_vec)

        def hessian(x_matrix, y_vec, theta_vec):
            hypo_vec = hypothesis_func(x_matrix, theta_vec)
            weight_vec = (1 - hypo_vec) * hypo_vec
            hessian_mat = np.matmul(np.transpose(x), weight_vec * x)
            return 1 / sample_size * hessian_mat

        def next_theta(x_matrix, y_vec, curr_theta_vec):
            gradient_vec = gradient(x_matrix, y_vec, curr_theta_vec)
            hessian_mat = hessian(x_matrix, y_vec, curr_theta_vec)
            return curr_theta_vec - self.step_size * np.matmul(np.linalg.inv(hessian_mat), gradient_vec)

        def theta_change_rate(theta_vec_1, theta_vec_2):
            return np.linalg.norm(theta_vec_1 - theta_vec_2, 1)

        if self.theta is None:
            self.theta = np.zeros(shape=(sample_dim, 1))

        theta_vec_pointer = self.theta.copy()
        next_theta_vec = next_theta(x, y, theta_vec_pointer)

        while theta_change_rate(theta_vec_pointer, next_theta_vec) >= self.eps:
            theta_vec_pointer = next_theta_vec
            next_theta_vec = next_theta(x, y, theta_vec_pointer)

        self.theta = next_theta_vec

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.matmul(x, self.theta) >= 0
        # *** END CODE HERE ***


if __name__ == "__main__":
    main(*sample_info[0])
