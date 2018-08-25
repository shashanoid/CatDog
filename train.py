import network
import data

training_data, test_data = data.create_data()

net = network.Network([2500, 50, 2])
net.SGD(training_data, 30, 10, 2.0, test_data=test_data)