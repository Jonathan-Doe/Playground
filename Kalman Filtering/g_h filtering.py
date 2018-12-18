import numpy as np
import matplotlib.pyplot as plt


class GHFilter:

    @classmethod
    def filter(cls, measurements, x0, dx, g, h, dt):
        """Perform g h filtering"""
        x_est = x0
        estimations = [x_est]

        for measurement in measurements:

            prediction = x_est + dx * dt
            residual = measurement - prediction

            dx = dx + h * residual / dt
            x_est = prediction + g * residual
            estimations.append(x_est)

        return np.array(estimations)


if __name__ == '__main__':
    """
    measures = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6]
    weights = [160 + x for x in range(len(measures))]
    estimations = GHFilter.filter(measures, 160, 1, 0.6, 0.66, 1)

    plot_data = [estimations, weights, measures]
    labels = ['Filter', 'Weights', 'Measures']
    linestyles = ["-", '--', 'o']
    colors = ['blue', "black", 'black']
    for x,label, linestyle,color in zip(plot_data, labels, linestyles, colors):

        if linestyle == "o":
            plt.scatter(range(len(x)), x, label=label, marker=linestyle, color=color)
        else:
            plt.plot(x, label=label, linestyle=linestyle, color=color)

    plt.legend()
    plt.show() 
    """


    def gen_data(x0,dx,count,noise_factor, acceleration = 0):
        return np.array([x0 + (dx + acceleration * i) * i + np.random.randn() * noise_factor for i in range(count)])
    """
    # bad initial guess
    data = gen_data(10, 0, 20, 0, 2)
    filtered = GHFilter.filter(data, 10, 2, 0.2,0.02, 1)
    plt.scatter(range(len(data)), data, marker="o")
    plt.plot(filtered)
    plt.show()
    """

    np.random.seed(100)
    zs = gen_data(x0=5., dx=5., count=50, noise_factor=50)
    data1 = GHFilter.filter(zs, x0=0., dx=5., dt=1., g=0.1, h=0.01)
    data2 = GHFilter.filter(zs, x0=0., dx=5., dt=1., g=0.4, h=0.01)
    data3 = GHFilter.filter(zs, x0=0., dx=5., dt=1., g=0.8, h=0.01)

    plt.scatter(range(len(zs)), zs, color='k', marker='o')
    plt.plot(data1, label='g=0.1', marker='s', c='C0')
    plt.plot(data2, label='g=0.4', marker='v', c='C1')
    plt.plot(data3, label='g=0.8', c='C2')
    plt.legend(loc=4)
    plt.show()
