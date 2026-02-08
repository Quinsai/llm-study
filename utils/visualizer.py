import matplotlib.pyplot as plt

def draw_line_plot(data, label_x, label_y, title, label_line=' ', img_path=''):
    plt.plot(data, color='blue', label=label_line)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(img_path + title + '.png')
    plt.close()