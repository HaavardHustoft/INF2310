def equalize(filename, bits=8):
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import misc

    image = misc.imread(filename, flatten=True)

    m, n = image.shape
    N = m*n
    bins = 2**bits

    #Finn først det kumulative histogrammet (p(r))
    histogram = np.histogram(image, bins=bins)
    normalized_histogram = histogram[0]/N
    cumulative_histogram = np.zeros(bins)
    cumulative_histogram[0] = normalized_histogram[0]

    L = 2**bits

    for i in range(1,bins):
        cumulative_histogram[i] = cumulative_histogram[i-1] + normalized_histogram[i]
    
    s = image.copy()
    for i in range(n):
        for j in range(m):
            s[i][j] = round((L - 1)*cumulative_histogram[int(round(image[i][j]))])
    
    z_histogram = np.histogram(s,bins=bins)
    z_normalized = z_histogram[0]/N
    z_cumulative = np.zeros(bins)
    z_cumulative[0] = z_normalized[0]

    for i in range(1,bins):
        z_cumulative[i] = z_cumulative[i-1] + z_normalized[i]
    

    G = s.copy()

    for i in range(n):
        for j in range(m):
            G[i][j] = round((L - 1)*z_cumulative[int(round(s[i][j]))])


    plt.figure()
    plt.plot(z_cumulative)
    plt.savefig('z_cumulative.png')




    plt.figure()
    plt.hist([s.ravel(), image.ravel()], bins=bins, color=['b', 'r'])
    plt.legend(['New', 'Original'])
    plt.savefig('histogram.png')

    plt.figure()
    plt.imshow(s, cmap='gray')
    plt.savefig('new_image.png')

    plt.figure()
    plt.plot(cumulative_histogram)
    plt.savefig('cumulative.png')

    

    
    
    


if __name__ == "__main__":
    equalize('car.png')