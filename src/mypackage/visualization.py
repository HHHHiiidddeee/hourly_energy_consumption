import matplotlib.pyplot as plt

def plot_forecast(y_true, y_pred, timesteps, title):
    plt.figure(figsize=(12, 5))
    plt.plot(timesteps, y_true, label="True")
    plt.plot(timesteps, y_pred, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.title(title)
    plt.legend()
    plt.show()
