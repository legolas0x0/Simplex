import numpy as np
from matrix_solve import LU
class Simplex:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.m = b.shape[0]
        self.B = np.array([[i] for i in range(self.m)])
        self.x = np.array([[0] for i in range(self.n)], dtype='float64')
    def get_mininum_reduced_cost_variable(self):
        db = []
        min_reduced_cost = 0
        reduced_cost_variable =  -1

        for i in range(self.n):
            #Como pseudo custo de variável básica é zero, não calculamos
            if i not in self.B:
                #Calcula o vetor db para a variável i
                db = LU(self.A[:, self.B.T[0]], -1*self.A[:, i].T)
                #Calcula o pseudo custo e escolhe o menor
                rc = self.c[i, 0] + np.dot(self.c.T[0, tuple(self.B.T[0])], db)
                if(rc < min_reduced_cost):
                    min_reduced_cost = rc
                    reduced_cost_variable = i
        #Cria o d a partir do db
        d =  np.zeros([self.n, 1], dtype='float64')
        d.T[0, self.B.T[0]] = db
        #coloca a variável não básica que queremos incrementar 
        d[reduced_cost_variable] = 1
        return d, min_reduced_cost, reduced_cost_variable 
    
    def optimize(self):
        #Calcular ponto inicial dado B escolhido
        self.x.T[0, self.B.T[0]] = LU(self.A[:, self.B.T[0]], self.b.T[0])
        while(True):
            print(self.x)

            #Calculamos o vetor d, o valor do menor custo reduzido e a variável do menor custo
            d, min_reduced_cost, reduced_cost_variable = self.get_mininum_reduced_cost_variable()

            #Se estamos no ótimo acabamos o programa
            if(min_reduced_cost == 0):
                return self.x,np.dot(self.c.T[0], self.x)

            #Escolhe a variável básica que vai sair 
            basic_variable_to_leave = -1
            teta = float("inf")
            for i in self.B.T[0]:
                    teta_candidate = -self.x[i,0]/d[i, 0]
                    if(abs(d[i, 0])>0.00001 and self.x[i,0]>0.0001):
                        if(teta_candidate > 0 and teta_candidate < teta):
                            teta = teta_candidate
                            basic_variable_to_leave = i
            #Atualizamos o vetor x e as variáveis básicas
            self.x += teta*d
            self.B[basic_variable_to_leave, 0] = reduced_cost_variable

                
