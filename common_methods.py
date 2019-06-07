import numpy as np
import numpy.fft as fft


def pad_input_2(input, pad_by):
    """
    We increase the size of input for all j by pad_by on each side of the matrix
    by inserting values from the opposite side
    """
    mesh_size = input.shape[0]

    B = np.eye(mesh_size, dtype=np.float32)
    for i in range(pad_by):
        a = np.zeros(mesh_size, dtype=np.float32)
        a[mesh_size - i - 1] = 1
        B = np.concatenate(([a], B), axis=0)
    for i in range(pad_by):
        a = np.zeros(mesh_size, dtype=np.float32)
        a[i] = 1
        B = np.concatenate((B, [a]), axis=0)

    return B @ input @ B.T


## The following methods are pretty much the same as in the original PDE-Net ##


############ Initial value generator ############

def initgen(mesh_size, freq=3, boundary='Periodic'):
    # Default: (mesh_size, freq, boundary) = ([250, 250], 4, 'Periodic')
    if np.iterable(freq):
        return freq
    # 250x250 normally distributed variables IFFTed and FFTed:
    x = _initgen_periodic(mesh_size, freq=freq)
    x = x * 100
    if boundary.upper() == 'DIRICHLET':
        dim = x.ndim
        for i in range(dim):
            y = np.arange(mesh_size[i]) / mesh_size[i]
            y = y * (1 - y)
            s = np.ones(dim, dtype=np.int32)
            s[i] = mesh_size[i]
            y = np.reshape(y, s)
            x = x * y
        x = x[[slice(1, None), ] * dim]
        x = x * 16
    return x


###################################################


def _initgen_periodic(mesh_size, freq=3):
    # np.random.seed(50)
    # Default: (mesh_size, freq) = ([250, 250], 4)
    dim = len(mesh_size)
    # Default: 250x250-matrix of normally distributed variables
    x = np.random.randn(*mesh_size)
    coe = fft.ifftn(x)
    # set frequency of generated initial value
    # Array of random ints in [freq, 2*freq - 1]
    freqs = np.random.randint(freq, 2 * freq, size=[dim, ])
    # freqs = [10,10]
    for i in range(dim):
        perm = np.arange(dim, dtype=np.int32)
        perm[i] = 0
        perm[0] = i
        # Permutes for i = 1 and does nothing for i = 0.
        coe = coe.transpose(*perm)
        coe[freqs[i] + 1:-freqs[i]] = 0
        coe = coe.transpose(*perm)
    x = fft.fftn(coe)
    assert np.linalg.norm(x.imag) < 1e-8
    x = x.real
    return x
