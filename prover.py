from help_functions import *


class NaiveProver:
    def __init__(self, secret, public):
        """Initialize the prover with a secret.
        public is pair of tensors
        """
        self.A = secret[0]
        self.B = secret[1]
        self.M = secret[2]

        self.P = None
        self.Q = None
        self.R = None

        self.C = public
        self.D = new_tensor_from_tensor_and_isomorphism(self.C, self.A, self.B, self.M)

    def generate_commit(self):
        """
        construct C' with 3 random matrices P, Q, R 
        generate hash of C'
        send hash
        """
        self.P = random_invertible_matrix(n)
        self.Q = random_invertible_matrix(n)
        self.R = random_invertible_matrix(n)
        C_prime = new_tensor_from_tensor_and_isomorphism(self.C, self.P, self.Q, self.R)
        self.hashed_C_prime = hash_tensor(C_prime)
        return self.hashed_C_prime
        
        

    def generate_challenge_response(self, challenge):
        """Respond to a challenge from the verifier.
        challenge is either 0/1 
        if 0: give back (P,Q,R) (C -> C')
        if 1: give back (P,Q,R) \circ (A,B,C)^-1 (D -> C')
        """
        if challenge == 0:
            return self.P, self.Q, self.R
        elif challenge == 1:
            A_inv = self.A.inverse()
            B_inv = self.B.inverse()
            M_inv = self.M.inverse()
            return A_inv * self.P, B_inv*self.Q,  M_inv*self.R
        else:
            print("Invalid challenge")
        
    

class SmartProver:
    def __init__(self, secret, public):
        """Initialize the prover with a secret.
        public is pair of tensors
        """
        self.A = secret[0]
        self.B = secret[1]
        self.M = secret[2]

        self.u = None

        self.C = public
        self.D = new_tensor_from_tensor_and_isomorphism(self.C, self.A, self.B, self.M)

    def generate_commit(self):
        """
        construct C' with the technique
        generate hash of C'
        send hash
        """
        u = find_corank1_pt_for_tensor(self.C)
        self.u = u
        try:
            LU, LV, LW = corank_1_to_3vector_tuples(u, self.C)
        except ValueError as e:
            print("Error in corank_1_to_3vector_tuples:", e)
            return None
        try:
            f, g, h = create_fgh(create_phi_head(LU, LV, LW, Tensor2.GaloisTensorMap(self.C)))
        except ZeroDivisionError as e:
            print("Error in create_fgh:", e)
            return None
        C_prime = new_tensor_from_tensor_and_isomorphism(self.C, LU * diagonal_matrix(f), LV * diagonal_matrix(g), LW * diagonal_matrix(h))

        self.hashed_C_prime = hash_tensor(C_prime)
        return self.hashed_C_prime

    def generate_challenge_response(self, challenge):
        """Respond to a challenge from the verifier.
        challenge is either 0/1 
        if 0: give back u (C -> C')
        if 1: give back A^-1 (D -> C')
        """
        if challenge == 0:
            return self.u
        elif challenge == 1:
            A_inv = self.A.inverse()
            return A_inv*self.u
        else:
            print("Invalid challenge")
        

