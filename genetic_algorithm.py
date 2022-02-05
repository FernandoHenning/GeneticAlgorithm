import random
from random import choices, choice
import math
from sympy import var
from sympy import sympify
from sympy.utilities.lambdify import lambdify
from statistics import mean

"""
    INITIAL PARAMETERS:
    - References resolution.
    - Function
    - Interval for x1, x2 and y1, y2. [min, max] [min, max]
    - Limit of generations
    - Limit of population
    - Initial population size
    
    OPTIONALS PARAMETERS
    - Probability of mutation.
    - Probability of reproduction.
    - Pruning probability
"""


class GeneticAlgorithm:

    def __init__(self,
                 resolution_x: float,
                 resolution_y: float,
                 interval_x: tuple,
                 interval_y: tuple,
                 limit_generations: int,
                 limit_population: int,
                 initial_populatin_size: int,
                 mutation_individual_prob: float,
                 mutation_gene_prob: float
                 ):
        # --------------
        x = var('x')
        y = var('y')
        expr = sympify('5/(y+sqrt(x))')
        self.f = lambdify((x, y), expr)
        # -----------------
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.interval_x = interval_x
        self.interval_y = interval_y
        self.limit_generations = limit_generations
        self.limit_population = limit_population
        self.initial_population_size = initial_populatin_size

        self.Rx = self.interval_x[1] - self.interval_x[0]
        self.Ry = self.interval_y[1] - self.interval_y[0]

        self.nPx = math.ceil(self.Rx / self.resolution_x) + 1
        self.nPy = math.ceil(self.Ry / self.resolution_y) + 1

        self.nBx = len(bin(self.nPx)) - 2
        self.nBy = len(bin(self.nPy)) - 2

        self.interval_i = (0, self.nPx - 1)
        self.interval_j = (0, self.nPy - 1)

        self.population = []
        self.best_cases = []
        self.worst_cases = []
        self.avg_cases = []

        self.mutation_individual_prob = mutation_individual_prob
        self.mutation_gene_prob = mutation_gene_prob

        self.first_generation = []

    def mutate(self, individual):

        p = random.random()
        if p < self.mutation_individual_prob:
            for _ in range(self.nBx):
                index = random.randrange(self.nBx)
                individual[0][index] = individual[0][index] if random.random() > self.mutation_gene_prob else \
                    abs(individual[0][index] - 1)
            for _ in range(self.nBy):
                index = random.randrange(self.nBy)
                individual[1][index] = individual[1][index] if random.random() > self.mutation_gene_prob else \
                    abs(individual[1][index] - 1)
            individual = self.generate_individual(individual[0], individual[1])
            return individual
        else:
            return individual

    def generate_individual(self, genotype_x, genotype_y):
        i = int("".join([str(i) for i in genotype_x]), 2)
        j = int("".join([str(i) for i in genotype_y]), 2)
        phenotype_xi = self.interval_x[0] + (i * self.resolution_x)
        phenotype_yj = self.interval_y[0] + (j * self.resolution_y)

        if phenotype_xi > self.interval_x[-1]:
            phenotype_xi = self.interval_x[-1]

        if phenotype_yj > self.interval_y[-1]:
            phenotype_yj = self.interval_y[-1]

        fitness = self.f(phenotype_xi, phenotype_yj)

        return [genotype_x, genotype_y, i, j, phenotype_xi, phenotype_yj, fitness]

    def pruning(self):
        self.population = self.population[:self.limit_population]

    def random_crossover(self, a, b):
        px = random.randint(1, self.nBx)
        py = random.randint(1, self.nBy)
        genotype_xa, genotype_ya = a[0][0:px] + b[0][px:], a[1][0:py] + b[1][py:]
        genotype_xb, genotype_yb = b[0][0:px] + a[0][px:], b[1][0:py] + a[1][py:]
        offspring_a = self.generate_individual(genotype_xa, genotype_ya)
        offspring_b = self.generate_individual(genotype_xb, genotype_yb)
        return offspring_a, offspring_b

    @staticmethod
    def select_parent(population):
        parents = []
        for _ in range(2):
            parents.append(choice(population))
        return parents

    def generate_initial_population(self):
        for i in range(self.initial_population_size):
            while True:
                genotype_x = choices([0, 1], k=self.nBx)
                genotype_y = choices([0, 1], k=self.nBy)
                individual = self.generate_individual(genotype_x, genotype_y)
                if self.interval_i[0] <= individual[2] <= self.interval_i[1] and\
                        self.interval_j[0] <= individual[3] <= self.interval_j[1]:
                    self.population.append(individual)
                    break
            # [[GENOTYPE X], [GENOTYPE Y], i, j, [PHENOTYPE I], [PHENOTYPE J], fitness]

    def run(self, minimize: bool):
        generation = 0

        self.generate_initial_population()
        self.population = sorted(
            self.population,
            key=lambda y: [x[6] for x in self.population],
            reverse=minimize
        )

        self.first_generation = self.population.copy()
        for i in range(self.limit_generations):
            # SORTED BY BETTER FITNESS
            for j in range(int(len(self.population) / 2)):
                parent = self.select_parent(self.population)
                offspring_a, offspring_b = self.random_crossover(parent[0], parent[1])
                offspring_a = self.mutate(offspring_a)
                offspring_b = self.mutate(offspring_b)
                self.population.append(offspring_a)
                self.population.append(offspring_b)
            self.population = sorted(
                self.population,
                key=lambda y: [y[6] for _ in self.population],
                reverse=minimize
            )
            self.best_cases.append(self.population[0])
            self.avg_cases.append(mean([x[6] for x in self.population]))
            self.worst_cases.append(self.population[-1])

            if len(self.population) > self.limit_population:
                self.pruning()
            generation += 1


"""
            x = []
            y = []
            for individual in self.population:
                x.append(individual[4])
                y.append(individual[5])

            colors = np.random.uniform(15, 80, len(x))
            # plot
            fig, ax = plt.subplots()
            ax.scatter(x, y, c=colors, vmin=0, vmax=100)

            ax.set(xlim=(self.interval_x[0], self.interval_x[1]), xticks=np.arange(0, self.interval_x[1]),
                   ylim=(self.interval_y[0], self.interval_y[1]), yticks=np.arange(0, self.interval_x[1]))
            plt.title(f"Generación {generation}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"images/generation {generation}.png")
"""