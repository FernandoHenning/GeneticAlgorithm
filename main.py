import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from genetic_algorithm import GeneticAlgorithm
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from tkinter import messagebox

# root window
root = tk.Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title('Algorítmo Genético - Maximizar y Minimizar')

root.configure(bg="white")

# configure the grid
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=7)

title_label = ttk.Label(root, text="Parámetros inciales del algoritmo", background='#fff',
                        font=('Lucida Sands', '12', 'bold'))
title_label.grid(column=0, row=0, )

# MÁXIMO X --------------------------------------------------------
max_x_label = ttk.Label(root, text="Valor máximo para X:", background='#fff', font=('Lucida Sands', '10'))

max_x_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

max_x_entry = ttk.Entry(root)
max_x_entry.insert(0, "20")
max_x_entry.grid(column=0, row=1, sticky=tk.E, padx=5, pady=5)

# MÍNIMO X---------------------------------------------------------
min_x_label = ttk.Label(root, text="Valor mínimo para X:", background='#fff', font=('Lucida Sands', '10'))

min_x_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

min_x_entry = ttk.Entry(root)
min_x_entry.insert(0, "0")
min_x_entry.grid(column=0, row=2, sticky=tk.E, padx=5, pady=5)

ttk.Separator(root, orient=tk.HORIZONTAL).grid(column=0, row=3, sticky="ew")

# MÁXIMO Y --------------------------------------------------------
max_y_label = ttk.Label(root, text="Valor máximo para Y:", background='#fff', font=('Lucida Sands', '10'))

max_y_label.grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)

max_y_entry = ttk.Entry(root)
max_y_entry.insert(0, "20")
max_y_entry.grid(column=0, row=4, sticky=tk.E, padx=5, pady=5)

# MÁXIMO Y --------------------------------------------------------
min_y_label = ttk.Label(root, text="Valor mínimo para Y:", background='#fff', font=('Lucida Sands', '10'))

min_y_label.grid(column=0, row=5, sticky=tk.W, padx=5, pady=5)

min_y_entry = ttk.Entry(root)
min_y_entry.insert(0, "10")
min_y_entry.grid(column=0, row=5, sticky=tk.E, padx=5, pady=5)

ttk.Separator(root, orient=tk.HORIZONTAL).grid(column=0, row=6, sticky="ew")

# RESOLUTION X --------------------------------------------------------
resolution_x_label = ttk.Label(root, text="Resolución para X:", background='#fff', font=('Lucida Sands', '10'))

resolution_x_label.grid(column=0, row=7, sticky=tk.W, padx=5, pady=5)

resolution_x_entry = ttk.Entry(root)
resolution_x_entry.insert(0, "0.01")
resolution_x_entry.grid(column=0, row=7, sticky=tk.E, padx=5, pady=5)

# RESOLUTION Y --------------------------------------------------------
resolution_y_label = ttk.Label(root, text="Resolución para Y:", background='#fff', font=('Lucida Sands', '10'))

resolution_y_label.grid(column=0, row=8, sticky=tk.W, padx=5, pady=5)

resolution_y_entry = ttk.Entry(root)
resolution_y_entry.insert(0, "0.01")
resolution_y_entry.grid(column=0, row=8, sticky=tk.E, padx=5, pady=5)

ttk.Separator(root, orient=tk.HORIZONTAL).grid(column=0, row=9, sticky="ew")
# MAX GENERATIONS -----------------------------------------------------
max_generations_label = ttk.Label(root, text="Límite de generaciones:", background='#fff', font=('Lucida Sands', '10'))

max_generations_label.grid(column=0, row=10, sticky=tk.W, padx=5, pady=5)

max_generations_entry = ttk.Entry(root)
max_generations_entry.insert(0, "1000")
max_generations_entry.grid(column=0, row=10, sticky=tk.E, padx=5, pady=5)

# INITIAL POPULATION -----------------------------------------------------
initial_population_label = ttk.Label(root, text="Población inicial:", background='#fff', font=('Lucida Sands', '10'))

initial_population_label.grid(column=0, row=11, sticky=tk.W, padx=5, pady=5)

initial_population_entry = ttk.Entry(root)
initial_population_entry.insert(0, "8")
initial_population_entry.grid(column=0, row=11, sticky=tk.E, padx=5, pady=5)

# MAX POPULATION -----------------------------------------------------
max_population_label = ttk.Label(root, text="Población máxima:", background='#fff', font=('Lucida Sands', '10'))

max_population_label.grid(column=0, row=12, sticky=tk.W, padx=5, pady=5)

max_population_entry = ttk.Entry(root)
max_population_entry.insert(0, "100")
max_population_entry.grid(column=0, row=12, sticky=tk.E, padx=5, pady=5)

# INDIVIDUAL MUTATION PROB -----------------------------------------------------
individual_mutation_prob_label = ttk.Label(root, text="Probabilidad de motación de individuos:", background='#fff',
                                           font=('Lucida Sands', '10'))

individual_mutation_prob_label.grid(column=0, row=13, sticky=tk.W, padx=5, pady=5)

individual_mutation_prob_entry = ttk.Entry(root)
individual_mutation_prob_entry.insert(0, "0.25")
individual_mutation_prob_entry.grid(column=0, row=13, sticky=tk.E, padx=5, pady=5)

# GENE MUTATION PROB -----------------------------------------------------
gen_mutation_prob_label = ttk.Label(root, text="Probabilidad de motación de genes:", background='#fff',
                                    font=('Lucida Sands', '10'))

gen_mutation_prob_label.grid(column=0, row=14, sticky=tk.W, padx=5, pady=5)

gen_mutation_prob_entry = ttk.Entry(root)
gen_mutation_prob_entry.insert(0, "0.15")
gen_mutation_prob_entry.grid(column=0, row=14, sticky=tk.E, padx=5, pady=5)


# BUTTONS ---------------------------------------------------------
def run(minimize: bool):
    if int((min_x_entry.get())) >= 0 and int(max_x_entry.get()) >= 0:
        ga = GeneticAlgorithm(float(resolution_x_entry.get()), float(resolution_y_entry.get()),
                              (float(min_x_entry.get()), float(max_x_entry.get())),
                              (float(min_y_entry.get()), float(max_y_entry.get())),
                              int(max_generations_entry.get()),
                              int(max_population_entry.get()),
                              int(initial_population_entry.get()),
                              float(individual_mutation_prob_entry.get()),
                              float(gen_mutation_prob_entry.get()))
        ga.run(minimize)

        figure1 = plt.figure()
        ax = plt.axes(projection='3d')
        x1 = np.arange(ga.interval_x[0], ga.interval_x[1], 0.1)
        y1 = np.arange(ga.interval_y[0], ga.interval_y[1], 0.1)
        x1, y1 = np.meshgrid(x1, y1)
        z1 = ga.f(x1, y1)
        ax.plot_surface(x1, y1, z1, color="red")
        # Showing the above plot
        ax.scatter(ga.population[0][4], ga.population[0][5], ga.population[0][6])
        ax.set_xlabel("X")

        ax.set_ylabel("Y")
        ax.set_zlabel("Aptitud")
        ax.set_title("Mejor individuo generado")
        ax.plot3D(0, 0, 0, 'green')

        individuo3d = FigureCanvasTkAgg(figure1, root)
        individuo3d.get_tk_widget().grid(column=1, row=0, rowspan=13, sticky=tk.W, padx=5, pady=5)

        figure2 = plt.figure()
        plt.plot(np.arange(0, ga.limit_generations), [x[6] for x in ga.best_cases], label="Best cases")
        plt.plot(np.arange(0, ga.limit_generations), [x[6] for x in ga.worst_cases], label="Worst cases")
        plt.plot(np.arange(0, ga.limit_generations), ga.avg_cases, label="Average cases")
        plt.legend()
        plt.title("Evolución de la población")
        plt.xlabel("Generaciones/Iteraciones")
        plt.ylabel("Valor de aptitud")
        linear = FigureCanvasTkAgg(figure2, root)
        linear.get_tk_widget().grid(column=2, row=0, rowspan=13, sticky=tk.W, padx=5, pady=5)


        figure3 = plt.figure()
        x = np.linspace(ga.interval_x[0], ga.interval_x[1])
        y = np.linspace(ga.interval_y[0], ga.interval_y[1])
        X, Y = np.meshgrid(x, y)

        Z1 = ga.f(X, Y)
        z = 50 * Z1
        z[:5, :5] = -1
        z = ma.masked_where(z <= 0, z)

        cp = plt.contourf(X, Y, z)
        cbar = plt.colorbar(cp)
        plt.title(f'Primer generación - tamaño de la población: {len(ga.first_generation)}')
        for individual in ga.first_generation:
            plt.scatter(individual[4], individual[5], marker='o', c='r', s=50)
        plt.xlabel('X')
        plt.ylabel('Y')
        first_generation = FigureCanvasTkAgg(figure3, root)
        first_generation.get_tk_widget().grid(column=2, row=14, rowspan=13, sticky=tk.W, padx=5, pady=5)

        figure4 = plt.figure()
        x = np.linspace(ga.interval_x[0], ga.interval_x[1])
        y = np.linspace(ga.interval_y[0], ga.interval_y[1])
        X, Y = np.meshgrid(x, y)

        Z1 = ga.f(X,Y)
        z = 50 * Z1
        z[:5, :5] = -1
        z = ma.masked_where(z <= 0, z)

        cp = plt.contourf(X, Y, z)
        plt.title(f'Última generación - tamaño de la población: {len(ga.population)}')
        for individual in ga.population:
            plt.scatter(individual[4], individual[5],marker='o',c='r',s=50)
        plt.xlabel('X')
        plt.ylabel('Y')
        last_generation = FigureCanvasTkAgg(figure4, root)
        last_generation.get_tk_widget().grid(column=1, row=14, rowspan=13, sticky=tk.W, padx=5, pady=5)
        messagebox.showinfo(
            message=f"Genotipo X: {ga.population[0][0]}\nGenotipo Y:{ga.population[0][1]}\ni:{ga.population[0][2]}, j: {ga.population[0][3]}, Fenotipo i: {ga.population[0][4]}, Fenotipo j: {ga.population[0][5]}, Aptitud: {ga.population[0][6]}",
            title="Mejor individuo")
        plt.show()
    else:
        messagebox.showerror(title="Valores fuera del dominio de la función", message=f"El valor {min_x_entry.get()} o {max_x_entry.get()} se encuentra fuera del dominio de la función.")


login_button = ttk.Button(root, text="Maximizar", command=lambda: run(True))
login_button.grid(column=0, row=15, sticky=tk.W, padx=5, pady=5)
login_button = ttk.Button(root, text="Minimizar", command=lambda: run(False))
login_button.grid(column=0, row=15, sticky=tk.E, padx=5, pady=5)

root.mainloop()
