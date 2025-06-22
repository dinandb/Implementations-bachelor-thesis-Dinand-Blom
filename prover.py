from help_functions import *


class NaiveProver:
    def __init__(self, secret, public):
        """Initialize the prover with a secret.
        public is pair of tensors
        """
        self.A = secret[0]
        self.B = secret[1]
        self.M = secret[2]

        self.C = public
        self.D = new_tensor_from_tensor_and_isomorphism(self.C, self.A, self.B, self.M)

    def generate_challenge():
        """
        construct C' with the technique
        generate hash of C'
        send hash
        """
        
        raise NotImplementedError
    def generate_challenge_naive():
        """
        construct C' with the technique
        generate hash of C'
        send hash
        """
        raise NotImplementedError

    def generate_challenge_response(self, challenge):
        """Respond to a challenge from the verifier.
        challenge is either 0/1 
        if 0: give back u (C -> C')
        if 1: give back A^-1 (D -> C')
        """
        raise NotImplementedError
    

class SmartProver:
    def __init__(self, secret, public):
        """Initialize the prover with a secret.
        public is pair of tensors
        """
        self.A = secret[0]
        self.B = secret[1]
        self.M = secret[2]

        self.C = public
        self.D = new_tensor_from_tensor_and_isomorphism(self.C, self.A, self.B, self.M)

    def generate_challenge():
        """
        construct C' with the technique
        generate hash of C'
        send hash
        """
        
        raise NotImplementedError
    def generate_challenge_naive():
        """
        construct C' with the technique
        generate hash of C'
        send hash
        """
        raise NotImplementedError

    def generate_challenge_response(self, challenge):
        """Respond to a challenge from the verifier.
        challenge is either 0/1 
        if 0: give back u (C -> C')
        if 1: give back A^-1 (D -> C')
        """
        raise NotImplementedError

