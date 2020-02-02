"""Package executor."""
import sys

from .runner import predict


def main(args=None):
    """Execute package."""
    if args is None:
        args = sys.argv[1:]

    ridge_reg_score, sklearn_score = predict()
    print(f'Mean Absolute Percentage Error of custom model on test set: {ridge_reg_score}%')
    print(f'Mean Absolute Percentage Error of sklearn model on test set: {sklearn_score}%')


if __name__ == "__main__":
    main()
