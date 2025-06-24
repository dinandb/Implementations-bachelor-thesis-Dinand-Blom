

import numpy as np

from sage.all import GF, vector, Matrix

"""
Tensor object, net zoals Tensor.py, maar zonder overbodige functies, cleaner in het algemeen en alles gebeurt in finite field.
Het vinden van corank1 punt gaat op dezelfde manier als voorheen:

Generate random tensor phi (nxnxn), vector b (n). Run find_zeros. Als phi.b niet corank1 heeft, probeer volgende (catch exception)
Totdat we phi, b hebben zodat phi.b corank1 heeft. Dan, op de unieke wijze het pad construeren. Door steeds de zeros te vinden van de nieuwe matrix.
Het vinden van zeroes moet ook gebeuren in finite field.
"""


class GaloisTensorMap:
    def __init__(self, tensor: np.ndarray):

        self.tensor = tensor
        

    def find_v_for_zero_form(self, u: np.ndarray, index : int, 
                            check_corank: bool = True, debugging=False, std_dir=True) -> np.ndarray:
        """
        
        """
        
        debugging = False
        if debugging:
            print(f"index: {index}")
            print(f"std dir: {std_dir}")
        # oude:
        # if index == 0:
            
        #     M = self.field(np.tensordot(self.tensor, u, axes = (0, 0))%self.q) # matrix over assen (1,2)
        # elif index == 1:
        #     M = self.field(np.tensordot(self.tensor, u, axes = (1, 0))%self.q) # matrix over assen (0,2)
        #     M = M.T                                                            # matrix over assen (2, 0)
        # elif index == 2:
        #     M = self.field(np.tensordot(self.tensor, u, axes = (2, 0))%self.q) # matrix over assen (0,1)
        if std_dir:
            if index == 0:
                
                M = (np.tensordot(self.tensor, u, axes = (0, 0))) # matrix over assen (1,2)
            elif index == 1:
                M = (np.tensordot(self.tensor, u, axes = (1, 0))) # matrix over assen (0,2)
                M = M.T                                                            # matrix over assen (2, 0)
            elif index == 2:
                M = (np.tensordot(self.tensor, u, axes = (2, 0))) # matrix over assen (0,1)
            else:
                raise ValueError("index not \in {0,1,2}")
        else:
            if index == 0:
                
                M = (np.tensordot(self.tensor, u, axes = (0, 0))) # matrix over assen (1,2)
                M = M.T # matrix over assen (2, 1)
            elif index == 1:
                M = (np.tensordot(self.tensor, u, axes = (1, 0))) # matrix over assen (0,2)
                
            elif index == 2:
                M = (np.tensordot(self.tensor, u, axes = (2, 0))) # matrix over assen (0,1)
                M = M.T # matrix over assen (1,0)
                
            else:
                raise ValueError("index not \in {0,1,2}")
        M = Matrix(M)
        # print(f"u: {u}")
        if debugging:
            print(f"tensor . u = {M}")
        
        max_rank = min(M.nrows(), M.ncols())
        
        if check_corank:
            rank = M.rank()
            if rank != max_rank - 1:
                raise ValueError(f"Matrix must have corank 1, but has rank {rank} out of {max_rank}")
            
        null = M.right_kernel().basis_matrix()
        # print(null)

        v = null[0]

        return v
    
    def evaluate (self, x = None, y = None, z = None, axes=None):
        # print(f"going to calculate:\n {self.tensor} * {vectors}")
        # print(f"first vector: {vectors[0]}")
        # print(f"axes = {axes}")


        # if axes is None:
        #     axes = [0]

        # if len(axes) == 1:
        #     axes *= len(vectors)
        # else:
        #     if len(axes) != len(vectors):
        #         print(f"len axes = {len(axes)}, but len vectors = {len(vectors)}")
        #         raise ValueError
        # assert len(vectors) == len(axes)
        # assert len(vectors) < 4
        
        # for i in range(len(vectors)):
        #         result = np.tensordot(result, vectors[i], axes=(axes[i],0))
        # # print("multiplied succesfully")
        # if result.shape == ():
        #     return result.item()
        # return result
        
        result = self.tensor 
        if x is not None:
            result = np.tensordot(result, x, axes=(0, 0))  # Contract along the first dimension 
        if y is not None:                                  # misschien andere axis volgorde
            result = np.tensordot(result, y, axes=(0, 0))  # Contract along the second dimension
        if z is not None:
            result = np.tensordot(result, z, axes=(0, 0))  # Contract along the third dimension
        if result.shape == ():
            return result.item()
        return result
    
    def evaluate2 (self, *vectors, axes=None):
        # print(f"going to calculate:\n {self.tensor} * {vectors}")
        # print(f"first vector: {vectors[0]}")
        # print(f"axes = {axes}")


        if axes is None:
            axes = [0]

        if len(axes) == 1:
            axes *= len(vectors)
        else:
            if len(axes) != len(vectors):
                print(f"len axes = {len(axes    )}, but len vectors = {len(vectors)}")
                raise ValueError
        assert len(vectors) == len(axes)
        assert len(vectors) < 4
        result = self.tensor
        for i in range(len(vectors)):
                result = np.tensordot(result, vectors[i], axes=(axes[i],0))
        # print("multiplied succesfully")
        if result.shape == ():
            return result.item()
        return result
