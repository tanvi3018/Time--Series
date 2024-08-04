import logging
from data_loader import load_data
from data_splitter import split_data
from model_trainer import train_model
from model_evaluator import evaluate_model

logging.basicConfig(level=logging.INFO)

def main():
    try:
        X, y = load_data('data.csv')
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_model(X_train, y_train)
        mse = evaluate_model(model, X_test, y_test)
        logging.info(f'Mean Squared Error: {mse}')
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == '__main__':
    main()
