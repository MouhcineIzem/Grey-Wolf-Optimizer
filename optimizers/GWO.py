

import random
import numpy
import math
from solution import solution
import time


def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # Max_iter=1000
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5

    # Initialiser alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialiser les positions de search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()


    print('GWO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):


            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])


            fitness = objf(Positions[i, :])

            # mettre a jour  Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness

                Alpha_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter)
        # a diminuer linearement  a de 2 a 0

        # mettre a jour la position de search agents et  omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a

                C1 = 2 * r2


                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])

                X1 = Alpha_pos[j] - A1 * D_alpha


                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a

                C2 = 2 * r2


                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])

                X2 = Beta_pos[j] - A2 * D_beta


                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a

                C3 = 2 * r2


                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])

                X3 = Delta_pos[j] - A3 * D_delta


                Positions[i, j] = (X1 + X2 + X3) / 3

        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    s.objfname = objf.__name__

    return s
