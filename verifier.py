
from help_functions import *
import random
class NaiveVerifier:
    def __init__(self, C, D):
        """Initialize the verifier with public information.
        this is nothing"""
        self.transformation_start_tensor = None
        self.hashed_C_prime = None
        self.C = C
        self.D = D

    def accept_and_generate_challenge(self, hashed_C_prime):

        """takes the hash of C_prime, save it
        Generate a challenge for the prover.
        -> either 0/1, save either 0/1, before this it should be None, after the protocol set to None (in verify)
        """
        self.hashed_C_prime = hashed_C_prime
        self.transformation_start_tensor = random.randint(0,1)
        return self.transformation_start_tensor


    def verify_response(self, response):
        """Verify the response from the prover.
        response is the transformation matrix
        if set to 0, transform from C, else from D. hash the resulting tensor and check if it matches hashed_C_prime which we saved
        returns True/False
        """
        if self.transformation_start_tensor == 0:
            C_prime = new_tensor_from_tensor_and_isomorphism(self.C, response[0], response[1], response[2])
        elif self.transformation_start_tensor == 1:
            C_prime = new_tensor_from_tensor_and_isomorphism(self.D, response[0], response[1], response[2])

        new_hashed_C_prime = hash_tensor(C_prime)

        return new_hashed_C_prime == self.hashed_C_prime

class SmartVerifier:
    def __init__(self, C, D):
        """Initialize the verifier with public information.
        this is nothing"""
        self.transformation_start_tensor = None
        self.hashed_C_prime = None
        self.C = C
        self.D = D

    def accept_and_generate_challenge(self, hashed_C_prime):

        """takes the hash of C_prime, save it
        Generate a challenge for the prover.
        -> either 0/1, save either 0/1, before this it should be None, after the protocol set to None (in verify)
        """
        self.hashed_C_prime = hashed_C_prime
        self.transformation_start_tensor = random.randint(0,1)
        return self.transformation_start_tensor
        
        


    def verify_response(self, response):
        """Verify the response from the prover.
        response is the representing vector
        if set to 0, transform from C, else from D. hash the resulting tensor and check if it matches hashed_C_prime which we saved
        returns True/False
        """
        if self.transformation_start_tensor == 0:
            LU, LV, LW = corank_1_to_3vector_tuples(response, self.C)
            f, g, h = create_fgh(create_phi_head(LU, LV, LW, Tensor2.GaloisTensorMap(self.C)))
            C_prime = new_tensor_from_tensor_and_isomorphism(self.C, LU * diagonal_matrix(f), LV * diagonal_matrix(g), LW * diagonal_matrix(h))

        elif self.transformation_start_tensor == 1:
            LUD, LVD, LWD = corank_1_to_3vector_tuples(response, self.D)
            fD, gD, hD = create_fgh(create_phi_head(LUD, LVD, LWD, Tensor2.GaloisTensorMap(self.D)))
            C_prime = new_tensor_from_tensor_and_isomorphism(self.D, LUD * diagonal_matrix(fD), LVD * diagonal_matrix(gD), LWD * diagonal_matrix(hD))
        
        new_hashed_C_prime = hash_tensor(C_prime)
        return new_hashed_C_prime == self.hashed_C_prime
           
        
