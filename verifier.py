
from help_functions import *

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
        import random
        # return random.randint(0,1)
        return 0
        


    def verify_response(self, response):
        """Verify the response from the prover.
        response is the transformation matrix
        if set to 0, transform from C, else from D. hash the resulting tensor and check if it matches hashed_C_prime which we saved
        returns True/False
        """
        
        C_prime = new_tensor_from_tensor_and_isomorphism(self.C, response[0], response[1], response[2])
        

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
        import random
        # return random.randint(0,1)
        return 0
        


    def verify_response(self, response):
        """Verify the response from the prover.
        response is the transformation matrix
        if set to 0, transform from C, else from D. hash the resulting tensor and check if it matches hashed_C_prime which we saved
        returns True/False
        """
        
        raise NotImplementedError
        
