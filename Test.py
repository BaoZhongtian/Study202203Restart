import numpy
import matplotlib.pylab as plt
import os

if __name__ == '__main__':
    load_path = 'E:/ProjectData/NCLS/Beaver-AllData-Character-220312-125400-Result'
    total_data = []
    for filename in os.listdir(load_path)[:-2]:
        current_data = numpy.genfromtxt(fname=os.path.join(load_path, filename), dtype=float)
        total_data.append(current_data)
    plt.plot(total_data)
    plt.show()
    print(min(total_data))
    print(os.listdir(load_path)[numpy.argmin(total_data)])
