import torch
import numpy as np

def cascaded_paraunit_matrix(N, n_stages, gain_per_sample, sparsity=1, matrix=None):
    '''
    create paraunitary matrix from scascaded design 
    see Scattering in Feedback Delay Networks, IEEE TASLP
    args:
        N (int):    size of the paraunitary matrix
        n_stages (init):    number of stages
        sparsity (int): time-wise sparsity [1->Inf] = [dense->sparse]
        matrix (string/tensor): if string, one of ['Hadamard', 'Orthogonal']
                                if tensor, a NxNx(n_stages+1) scalar tensor
    '''
    K = n_stages+1
    if isinstance(matrix,str):
        U = torch.zeros((N,N,K))
        if matrix == 'Hadamard':
            U = torch.tensor(hadamard_matrix(N)).unsqueeze(-1).expand(N,N,K)
        elif matrix == 'Orthogonal':
            for k in range(K):
                U[:,:,k] = torch.tensor(random_orth_matrix(N))
        else:
            raise ValueError(
                'The type of matrix is none of [\'Hadamard\', \'Orthogonal\']')

    elif isinstance(matrix, torch.Tensor):
        U = matrix

    sparsity_vect = torch.ones((n_stages))
    sparsity_vect[0] = sparsity
    pulse_size = 1
    U = U.to(torch.float32)
    V = U[:,:,0]
    for k in range(1,K):

        # np.random.seed(130798)
        shift_L = shift_mat_distribute(V, sparsity_vect[k-1], pulse_size)

        G = torch.diag(gain_per_sample**shift_L).to(torch.float32)
        R = torch.matmul(U[:,:,k],G)

        V = shift_matrix(V, shift_L, direction='left')
        V = poly_matrix_conv(R, V)

        pulse_size = pulse_size * N*sparsity_vect[k-1]
    
    return V

def poly_matrix_conv(A, B):
    ''' Multiply two matrix polynomials A and B by convolution '''
    
    if len(A.shape) == 2:
        A = A.view(A.shape[0], A.shape[1], 1)
    if len(B.shape) == 2:
        B = B.view(B.shape[0], B.shape[1], 1)

    # Get the dimensions of A and B
    szA = A.shape
    szB = B.shape

    if szA[1] != szB[0]:
        raise ValueError('Invalid matrix dimension.')

    C = torch.zeros((szA[0], szB[1], szA[2] + szB[2] - 1))

    A = A.permute(2, 0, 1)
    B = B.permute(2, 0, 1)
    C = C.permute(2, 0, 1)

    for row in range(szA[0]):
        for col in range(szB[1]):
            for it in range(szA[1]):
                C[:, row, col] = C[:, row, col] + torch.conv1d(B[:, it, col].unsqueeze(0).unsqueeze(0), A[:, row, it].unsqueeze(0).unsqueeze(0))

    C = C.permute(1, 2, 0)

    return C

def shift_matrix(X, shift, direction='left'):
    ''' 
    shift in polynomial matrix in time-domain by shift samples
    direction is either Left or Right
    '''
    
    N = X.shape[0]
    # Find the last nonzero element indices along last dim
    if len(X.shape) == 2:
        X = X.view(X.shape[0], X.shape[1], 1)
    order = torch.max(torch.nonzero(X, as_tuple=True)[-1])
    if direction.lower() == 'left':
        required_space = order + shift.reshape(-1,1)
        additional_space = int((required_space.max() - X.shape[-1]) + 1)
        X = torch.cat((X, torch.zeros((N,N,additional_space))), dim=-1)
        for i in range(N):
            X[i, :, :] = torch.roll(X[i, :, :], shift[i].item(), dims=-1)
    elif direction.lower() == 'right':
        required_space = order + shift.reshape(1,-1)
        additional_space = int((required_space.max() - X.shape[-1]) + 1)
        X = torch.cat((X, torch.zeros((N,N,additional_space))), dim=-1)
        for i in range(N):
            X[:, i, :] = torch.roll(X[:, i, :], shift[i].item(), dims=-1)

    return X 

def shift_mat_distribute(X, sparsity, pulse_size):
    '''shift in polynomial matrix in time-domain such that they don't overlap'''
    N = X.shape[0]
    rand_shift = torch.floor(sparsity * (torch.arange(0,N) + torch.rand((N))*0.99))
    return (rand_shift * pulse_size).int()

def hadamard_matrix(N):
    '''Generate a hadamard matrix of size N'''
    X = np.array([[1]])
    # Create a Hadamard matrix of the specified order
    # TODO remove for loop becuase all matrices look the same
    while X.shape[0] < N:
        # Kronecker product to generate a larger Hadamard matrix
        X = np.kron(X, np.array([[1, 1], [1, -1]]))/np.sqrt(2)

    return X

def random_orth_matrix(N):
    '''Generate a random orthogonal matrix os size N'''

    A = np.random.randn(N, N)
    # Perform QR factorization
    Q, R = np.linalg.qr(A)
    # Modify Q by multiplying it with a diagonal matrix of signs from the diagonal of R
    diagonal_signs = np.sign(np.diag(R))
    Q = Q.dot(np.diag(diagonal_signs))
    return Q

if __name__ == '__main__':
    cascaded_paraunit_matrix(4, 3, gain_per_sample=0.9999, sparsity = 3, matrix='Hadamard')