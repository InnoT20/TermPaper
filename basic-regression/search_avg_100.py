import load_dataset
import network

train_data, train_labels, test_data, test_labels = load_dataset.load()

answers = []

for i in range(0, 100):
    model = network.Model(input_units=len(train_data.keys()), hidden_units=64, epochs=500)
    model.fit(train_data, train_labels)
    loss, mae, mse = model.evaluate(test_data, test_labels)
    # model.plot_history()
    print("Среднее абсолютное отклонение на проверочных данных: {:5.2f} галлон на милю".format(mae))
