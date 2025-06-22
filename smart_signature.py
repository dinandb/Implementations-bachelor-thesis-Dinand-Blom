from help_functions import *
from verifier import SmartVerifier
from prover import SmartProver

u, C = find_corank1_pt_T()
A = random_matrix(n)
B = random_matrix(n)
M = random_matrix(n)
D = new_tensor_from_tensor_and_isomorphism(C, A, B, M)
S = Tensor2.GaloisTensorMap(C)
T = Tensor2.GaloisTensorMap(D)
x = random_vector(n)
y = random_vector(n)
z = random_vector(n)

LU, LV, LW = corank_1_to_3vector_tuples(u, C)
LUD, LVD, LWD = corank_1_to_3vector_tuples(A.inverse()*u, D)

f,g,h = create_fgh(create_phi_head(LU,LV,LW,Tensor2.GaloisTensorMap(C)))
C_prime = new_tensor_from_tensor_and_isomorphism(C, LU*diagonal_matrix(f), LV*diagonal_matrix(g), LW*diagonal_matrix(h))
U = Tensor2.GaloisTensorMap(C_prime)

fD,gD,hD = create_fgh(create_phi_head(LUD,LVD,LWD,Tensor2.GaloisTensorMap(D)))

C_prime_hyp = new_tensor_from_tensor_and_isomorphism(D, LUD*diagonal_matrix(fD), LVD*diagonal_matrix(gD), LWD*diagonal_matrix(hD))


"""
uses the verifier and prover classes
"""

prover = None
verifier = None


def init():
    prover = SmartProver([random_matrix(n), random_matrix(n), random_matrix(n)], random_tensor(n))
    verifier

def one_iteration():
    pass


def main():
    init()
    one_iteration()
