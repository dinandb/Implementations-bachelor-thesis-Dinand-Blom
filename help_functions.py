import Tensor2
import numpy as np
from sage.all import GF, vector, Matrix, diagonal_matrix
import hashlib

q = 8191
n = 13


F = GF(q)

def is_in_span(A, v):
    # A is a Matrix object, v is vector object, both in field
    # returns true iff v is in span A, ie exists a vector x such that Ax = v
    try:
        x = A.solve_right(v)
    except ValueError as e:
        x = None
    # print(f"x:\n{x}")
    return x is not None # if x is not none, there is a solution meaning v \in span(A)

def random_vector(n):
    return vector(F, [F.random_element() for _ in range(n)])

def random_matrix(n):
    return Matrix([[F.random_element() for _ in range(n)] for _ in range(n)])

def random_invertible_matrix(n):
    while True:
        M = random_matrix(n)
        if M.is_invertible():
            return M

def random_tensor(n):
    return np.array([[[F.random_element() for _ in range(n)] for _ in range(n)] for _ in range(n)])


def hash_tensor(tensor):
    """Hashes a tensor over F_q using SHA-256."""
    tensor_bytes = tensor.astype(np.uint16).tobytes()  # Ensure consistent byte representation
    return hashlib.sha256(tensor_bytes).hexdigest()

def find_corank1_pt_T(q=q, n=n):
    m = k = n

    while True:
        # print("M is redefined")
        M = random_tensor(n)
        # print(M)
        T = Tensor2.GaloisTensorMap(M)
        for _ in range(q**2):
            isCorank1 = False
            u = random_vector(n)
            try:
                v = T.find_v_for_zero_form(u, 0) # v is kernelvector of T.u (1, 2) over axis 2
                
                if np.array_equal(v, np.zeros(n)):
                    raise ValueError
                
                isCorank1 = True
            
            except ValueError:
                # print(f"{u} was not corank1")
                pass

            if isCorank1:
                # print(f"{u} is corank1 voor tensor \n {M}")
                # print(f"norm u: {normalize(u, n)}")
                # print(f"found v: {v}")
                break
        
        if isCorank1:
            # print("corank1 point found")
            break
    return u, M


def new_tensor_from_tensor_and_isomorphism(C, A, B, M, n=n):
    # Apply the transformation to each matrix in the 3D tensor
    # D = np.array(
    #     [np.array(A * Matrix(c) * B) for c in C],
    #     dtype=object
    # )
    # print('hello')

    D = np.array([ np.array(sum([ A[i,j]*B.T*Matrix(C[i,:,:])*M for i in range(n)])) for j in range(n)], dtype=object).reshape(C.shape)    
    # D = np.array([ np.array(sum([ M[i,j]*A*Matrix(C[j,:,:])*B for j in range(n)])) for i in range(n)], dtype=object)
    # Reshape D to match the shape of C
    return D.reshape(C.shape)


def tensor_equality(C, D):
    # for c in C:
    #     if not matrix_in_tensor_span(Matrix(c), D):
    #         return False
    # return True
    assert len(C) == len(D)# == len(C[0]) == len(C[0][0]) == len(D[0]) == len(D[0][0])
    
    n = len(C)
    return not np.sum(np.array([[[1 if C[i][j][k]!=D[i][j][k] else 0 for k in range(n)] for j in range(n)] for i in range(n)]))

def find_corank1_pt_for_tensor(D,n=n):
    T = Tensor2.GaloisTensorMap(D)
    # for _ in range(q**2):
    isCorank1 = False
    while not isCorank1:
        
        u = random_vector(n)
        try:
            v = T.find_v_for_zero_form(u, 0) # v is kernelvector of T.u (1, 2) over axis 2
            
            if np.array_equal(v, np.zeros(n)):
                raise ValueError
            
            isCorank1 = True
        
        except ValueError:
            # print(f"{u} was not corank1")
            pass

        if isCorank1:
            pass
            # print(f"{u} is corank1 voor tensor \n {D}")
            # print(f"norm u: {normalize(u, n)}")
            # print(f"found v: {v}")
            
    return u

def corank_1_to_3vector_tuples(u, C, n=None, debugging=False, std_dir = True):
    std_dir = False
    if n is None:
        n = len(C)

    """
    m = n = k
    We view C as a (F^n_q)^3 -> F_q Tensor2 now
    So
    u \in F^n_q satisfies C(u, *, *) is of corank-1 as bilinear form (has rank n - 1)
    """
    

    L_U = [u]
    L_V = (np.empty((0, n), dtype=int))
    L_W = (np.empty((0, n), dtype=int))

    # debugging = True
    

    T = Tensor2.GaloisTensorMap(C)
    # T.find_v_for_zero_form(u, 0) # in this function it is checked if u is a corank1 point of T

    # this is also checked immediately in the loop 

    for i in range(1, n+1):
        u = L_U[-1]
        v = vector(F, T.find_v_for_zero_form(u, 0, debugging=debugging, std_dir=std_dir)) # 0 indicates that we are at index 0 and want to find a zero at index 0+1%3 = 1
        # v = vector(F, T.find_v_for_zero_form(u, 0, debugging=debugging, std_dir=std_dir)) # 0 indicates that we are at index 0 and want to find a zero at index 0+1%3 = 1
        if is_in_span(Matrix(L_V), v):
            raise Exception(f"Found v \n{v} \nwas already in span of L_V\n {L_V}")
        if debugging:
            print(f"adding v: {v}")
        L_V = np.vstack((L_V, v))
        
        index = 2 if std_dir else 1
        w = vector(F, T.find_v_for_zero_form(v, index, debugging=debugging, std_dir=std_dir)) # 1 indicates that we are at index 0 and want to find a zero at index 1+1%3 = 2
        # w = vector(F, T.find_v_for_zero_form(v, index, debugging=debugging, std_dir=std_dir), n) # 1 indicates that we are at index 0 and want to find a zero at index 1+1%3 = 2

        if is_in_span(Matrix(L_W), w):
            raise Exception(f"Found w \n{w} \nwas already in span of L_W\n {L_W}")
        if debugging:
            print(f"adding w: {w}")
        L_W = np.vstack((L_W, w))

        if i == n:
            break
        index = 1 if std_dir else 2
        u_next = vector(F, T.find_v_for_zero_form(w, index, debugging=debugging, std_dir=std_dir)) # 2 indicates that we are at index 0 and want to find a zero at index 2+1%3 = 0
        # u_next = vector(F, T.find_v_for_zero_form(w, index, debugging=debugging, std_dir=std_dir), n) # 2 indicates that we are at index 0 and want to find a zero at index 2+1%3 = 0

        if is_in_span(Matrix(L_U), u_next):
            raise Exception(f"Found u \n{u_next} \nwas already in span of L_U\n {L_U}")
        if debugging:
            print(f"adding u: {u_next}")
        L_U = np.vstack((L_U, u_next))

    T.find_v_for_zero_form(u_next, 0) # this will execute the find zero form, but not use the result
    # the reason we execute it is because the function contains a check to verify whether u_next is actually a corank-1 point of T

    return Matrix(L_U).T, Matrix(L_V).T, Matrix(L_W).T



def create_phi_head(L_U, L_V, L_W, phi): # as defined in chapter 4.3 of 
    # algorithms for solving MCE

    # phi is the Tensor2 of interest
    # L_U, L_V, L_W are the vector tuples created above

    # L_U = np.array(L_U) # niet transpose hier, is gechecked
    # L_V = np.array(L_V)
    # L_W = np.array(L_W) # ?? np.array

    return lambda x, y, z: phi.evaluate(L_U * x, L_V * y, L_W * z)

def e(i):
    # e = np.full(n, 0)
    e = vector(F, np.zeros(n))
    e[i] = 1
    return e

def create_fgh(phi_head, n=n):

    
    a = vector(F, ([phi_head(e(i), e(1), e(0)) for i in range(n)])) 
    b = vector(F, [phi_head(e(0), e(j), e(0)) for j in range(n)])  
    c = vector(F, [phi_head(e(0), e(1), e(k)) for k in range(n)])  
    d1 = phi_head(e(0), e(1), e(0))
    d2 = phi_head(e(1), e(2), e(4))
    d3 = phi_head(e(0), e(2), e(1))
    d4 = phi_head(e(1), e(0), e(1))

    

    # Initialize arrays for f_i, g_j, h_k
    f = vector(F, np.zeros(n))
    g = vector(F, np.zeros(n))
    h = vector(F, np.zeros(n))

    # Calculate f_1, g_1, h_1, f_2, g_2, h_2
    # Note that we have only 4 equations for f1,f2,g1,g2,h1,h2, so we get two degrees of freedom.
    # we set g2 = h1 = 1

    # g[1] = 1 # oude versie
    # h[0] = 1

    # # then the rest simplifies to:

    # f[0] = 1/d1
    # f[1] = (b[2]*c[4])/(d1**2 * d2)
    # h[1] = b[2]/(f[0]*d1*d3)
    # g[0] = 1/(f[1]*h[1]*d4)

    # # Calculate other elements (for indices >= 3)
    # for i in range(2, n):
    #     f[i] = d1 / a[i] * f[0] # f_i
    #     g[i] = d1 / b[i] * g[1] # g_j
    #     h[i] = d1 / c[i] * h[0] # h_k

    g[1] = 1
    h[0] = 1

    # then the rest simplifies to:

    # f[0] = 1/d1
    # f[0] = pow(d1, -1, q)
    f[0] = d1**-1
    # f[1] = (b[2]*c[4])/(d1**2 * d2)
    # f[1] = 1/(d1**2 * d2) * (b[2]*c[4])
    # f[1] = pow(d1**2 * d2, -1, q) * (b[2]*c[4])
    f[1] = (d1**2 * d2)**-1 * (b[2]*c[4])
    # h[1] = b[2]/(f[0]*d1*d3)
    # h[1] = 1/(f[0]*d1*d3) * b[2]
    # h[1] = pow(f[0]*d1*d3, -1, q) * b[2]
    # h[1] = (f[0]*d1*d3)**-1 * b[2]
    h[1] = b[2]/d3
    # g[0] = 1/(f[1]*h[1]*d4)
    # g[0] = pow((f[1]*h[1]*d4), -1, q)
    g[0] = (f[1]*h[1]*d4)**-1

    # Calculate other elements (for indices >= 3)
    for i in range(2, n):
        # f[i] = d1 / a[i] * f[0] # f_i
        # f[i] = d1 * (pow(a[i] * f[0], -1, q))
        
        f[i] = a[i]**-1
        # g[i] = d1 / b[i] * g[1] # g_j
        # g[i] = d1 * pow(b[i] * g[1], -1, q)
        g[i] = d1 * g[1] * b[i]**-1
        
        # h[i] = d1 / c[i] * h[0] # h_k
        # h[i] = d1 * pow(c[i] * h[0], -1, q) 
        h[i] = d1 * c[i]**-1

    # phi_bar = lambda x, y, z: phi_head(scaleVectorMult(vector(F, f), x), scaleVectorMult(vector(F, g), y), scaleVectorMult(vector(F, h), z))

    # print("henk")
    # print(f"phi bar (e1, e5,e1) {phi_bar(e(0), e(4), e(0))}")
    # print(f"f1g5h1 * phi head(e1, e5, e1) {f[0], g[4], h[0], phi_head(e(0), e(4), e(0))}")
    # print(f[0]* g[4]* h[0]* phi_head(e(0), e(4), e(0)))
    # print(f"f1d1g2h1 {f[0], d1, g[1], h[0]} res = {f[0]*d1*g[1]*h[0]}")

    # check done!
        

    return (vector(F, f)), (vector(F, g)), (vector(F, h))

def scaleVectorMult(v, w):
    return vector(F, [w[i] * v[i] for i in range(len(w))])


def create_phi_bar(phi_head, n=n):
    f, g, h = create_fgh(phi_head, n)
    return lambda x, y, z: phi_head(scaleVectorMult(f, x), scaleVectorMult(g, y), scaleVectorMult(h, z))

