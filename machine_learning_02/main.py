import tensorflow as tf
import tensorflow_datasets as tfds

import math
import random

tfds.disable_progress_bar()

principal_values = [
    5000,
    10000,
    20000,
    30000,
    40000,
    50000,
    100000,
    150000,
    200000,
    250000,
    350000,
    500000,
    750000,
    1000000,
]

interest_rate_values = [
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.050,
    0.100,
    0.125,
    0.150,
    0.200,
    0.250,
]

number_of_payments_values = [
    36,     # 3 years
    48,     # 4 years
    60,     # 5 years
    72,     # 6 years
    120,    # 10 years
    240,    # 20 years
    360,    # 30 years
    600,    # 50 years
]

def calculate_monthly_payment(principal: float, interest_rate: float, number_of_payments: float) -> float:
    return principal * (interest_rate * ((1.0 + interest_rate) ** number_of_payments) / ((1.0 + interest_rate) ** number_of_payments - 1.0))

def predict_monthly_payment(principal: float, interest_rate: float, number_of_payments: float, model: tf.keras.Sequential) -> float:
    return model.predict([(principal, interest_rate, number_of_payments)])[0][0]

def create_datasets() -> (list, list, list, list):
    train_data_features = []
    train_data_values = []
    
    test_data_features = []
    test_data_values = []

    for principal in principal_values:
        for interest_rate in interest_rate_values:
            for number_of_payments in number_of_payments_values:
                if random.random() < 0.9:
                    train_data_features.append((principal, interest_rate, number_of_payments))
                    train_data_values.append(calculate_monthly_payment(principal, interest_rate, number_of_payments))
                else:
                    test_data_features.append((principal, interest_rate, number_of_payments))
                    test_data_values.append(calculate_monthly_payment(principal, interest_rate, number_of_payments))

    return train_data_features, train_data_values, test_data_features, test_data_values

def main():
    print(f'Pricipal values = {len(principal_values)}')
    print(f'Interest rate values = {len(interest_rate_values)}')
    print(f'Number of payment values = {len(number_of_payments_values)}')
    print(f'Total combinations = {len(principal_values) * len(interest_rate_values) * len(number_of_payments_values)}')

    # Generate my datasets.
    train_data_features, train_data_values, test_data_features, test_data_values = create_datasets()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data_features, train_data_values))
    print(f'train_dataset: {train_dataset}')

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_features, test_data_values))
    print(f'test_dataset: {test_dataset}')

    # Randomize and batch my datasets.
    BATCH_SIZE = 32
    train_dataset_len = len(list(train_dataset.as_numpy_iterator()))
    test_dataset_len = len(list(test_dataset.as_numpy_iterator()))
    train_dataset = train_dataset.cache().repeat().shuffle(train_dataset_len).batch(BATCH_SIZE)
    test_dataset = test_dataset.cache().batch(BATCH_SIZE)

    # Create my polynomial model.
    model_linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=3, input_shape=[3,]),   # input layer
        tf.keras.layers.Dense(units=100),  # hidden layer
        tf.keras.layers.Dense(units=100),  # hidden layer
        tf.keras.layers.Dense(units=100),  # hidden layer
        tf.keras.layers.Dense(units=1) # output layer
    ])

    model_linear.compile(loss=tf.losses.MeanAbsolutePercentageError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=['accuracy'])

    # Create my non-linear model using ReLU.
    model_relu = tf.keras.Sequential([
        tf.keras.layers.Dense(units=3, input_shape=[3,]),   # input layer
        tf.keras.layers.Dense(units=100, activation=tf.nn.elu),  # hidden layer
        tf.keras.layers.Dense(units=100, activation=tf.nn.elu),  # hidden layer
        tf.keras.layers.Dense(units=100, activation=tf.nn.elu),  # hidden layer
        tf.keras.layers.Dense(units=1) # output layer
    ])

    model_relu.compile(loss=tf.losses.MeanAbsolutePercentageError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=['accuracy'])

    # Train my models.
    model_linear.fit(train_data_features, train_data_values, epochs=100)
    model_relu.fit(train_data_features, train_data_values, epochs=100)

    # Check the accuracy of the polynomial model with the test dataset.
    test_loss, test_accuracy = model_linear.evaluate(test_data_features, test_data_values)
    print('accuracy(model_linear):', test_accuracy)

    # Check the accuracy of the non-linear model with the test dataset.
    test_loss, test_accuracy = model_relu.evaluate(test_data_features, test_data_values)
    print('accuracy(model_relu):', test_accuracy)

if __name__ == "__main__":
    main()
