import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# Define the Rastrigin function
def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

# PSO Algorithm Implementation
class PSO_Params:
    def __init__(self, num_particles, dimensions, iterations):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.iterations = iterations
        self.w = 0.5    # Inertia weight
        self.c1 = 1.5   # Cognitive coefficient
        self.c2 = 1.5   # Social coefficient

def particle_swarm_optimization(rastrigin, params):
    num_particles = params.num_particles
    dimensions = params.dimensions
    iterations = params.iterations

    # Initialize positions and velocities
    positions = np.random.uniform(-5.12, 5.12, (num_particles, dimensions))
    velocities = np.zeros((num_particles, dimensions))
    personal_best = positions.copy()
    personal_best_scores = np.array([rastrigin(p) for p in positions])
    global_best_idx = np.argmin(personal_best_scores)
    global_best = personal_best[global_best_idx]
    global_best_score = personal_best_scores[global_best_idx]

    # Record position history and best scores over time
    history = [positions.copy()]
    best_score_history = [global_best_score]

    for i in range(iterations):
        r1 = np.random.rand(num_particles, dimensions)
        r2 = np.random.rand(num_particles, dimensions)

        # Update velocities
        velocities = (params.w * velocities + 
                      params.c1 * r1 * (personal_best - positions) +
                      params.c2 * r2 * (global_best - positions))

        # Update positions
        positions += velocities
        positions = np.clip(positions, -5.12, 5.12)

        # Evaluate fitness
        scores = np.array([rastrigin(p) for p in positions])

        # Update personal best positions
        better_mask = scores < personal_best_scores
        personal_best[better_mask] = positions[better_mask]
        personal_best_scores[better_mask] = scores[better_mask]

        # Update global best position
        current_global_best_idx = np.argmin(personal_best_scores)
        if personal_best_scores[current_global_best_idx] < global_best_score:
            global_best = personal_best[current_global_best_idx]
            global_best_score = personal_best_scores[current_global_best_idx]

        # Record positions and the best score
        history.append(positions.copy())
        best_score_history.append(global_best_score)

    return np.array(history), best_score_history

# DE Algorithm Implementation
class DE_Params:
    def __init__(self, pop_size, dimensions, iterations, F=0.8, CR=0.7):
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.iterations = iterations
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover rate

def differential_evolution(rastrigin, params):
    pop_size = params.pop_size
    dimensions = params.dimensions
    iterations = params.iterations
    F = params.F
    CR = params.CR

    # Initialize population
    population = np.random.uniform(-5.12, 5.12, (pop_size, dimensions))
    scores = np.array([rastrigin(ind) for ind in population])

    # Record position history
    history = [population.copy()]
    best_score_history = []

    best_idx = np.argmin(scores)
    best = population[best_idx].copy()
    best_score = scores[best_idx]
    best_score_history.append(best_score)

    for i in range(iterations):
        for j in range(pop_size):
            # Mutation
            idxs = [idx for idx in range(pop_size) if idx != j]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), -5.12, 5.12)

            # Crossover
            cross_points = np.random.rand(dimensions) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])

            # Selection
            f = rastrigin(trial)
            if f < scores[j]:
                population[j] = trial
                scores[j] = f

        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best = population[current_best_idx].copy()
            best_score = scores[current_best_idx]

        history.append(population.copy())
        best_score_history.append(best_score)

    return np.array(history), best_score_history

# SA Algorithm Implementation
class SA_Params:
    def __init__(self, initial_temp, final_temp, iterations, alpha=0.95):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.iterations = iterations
        self.alpha = alpha  # Cooling rate

def simulated_annealing(rastrigin, params):
    temp = params.initial_temp
    iterations = params.iterations
    alpha = params.alpha

    # Initialize
    current = np.random.uniform(-5.12, 5.12, 2)
    current_score = rastrigin(current)
    best = current.copy()
    best_score = current_score

    history = [current.copy()]
    best_score_history = [best_score]

    for i in range(iterations):
        # Generate neighbor
        neighbor = current + np.random.normal(0, 0.5, size=current.shape)
        neighbor = np.clip(neighbor, -5.12, 5.12)
        neighbor_score = rastrigin(neighbor)

        # Update decision based on probability
        delta = neighbor_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best = current.copy()
                best_score = current_score

        history.append(current.copy())
        best_score_history.append(best_score)

        # Decrease temperature
        temp *= alpha
        if temp < params.final_temp:
            break

    return np.array(history), best_score_history

# WOA Algorithm Implementation
class WOA_Params:
    def __init__(self, num_whales, dimensions, iterations, a=2):
        self.num_whales = num_whales
        self.dimensions = dimensions
        self.iterations = iterations
        self.a = a  # Parameter that decreases linearly from a=2 to a=0

def whale_optimization_algorithm(rastrigin, params):
    num_whales = params.num_whales
    dimensions = params.dimensions
    iterations = params.iterations
    a = params.a

    # Initialize population
    population = np.random.uniform(-5.12, 5.12, (num_whales, dimensions))
    scores = np.array([rastrigin(ind) for ind in population])

    # Find best
    best_idx = np.argmin(scores)
    best = population[best_idx].copy()
    best_score = scores[best_idx]

    history = [population.copy()]
    best_score_history = [best_score]

    for t in range(iterations):
        a_linear = a - t * (a / iterations)

        for i in range(num_whales):
            r = np.random.rand()
            A = 2 * a_linear * r - a_linear
            C = 2 * np.random.rand()

            p = np.random.rand()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best - population[i])
                    population[i] = best - A * D
                else:
                    rand_whale = population[np.random.randint(0, num_whales)]
                    D = abs(C * rand_whale - population[i])
                    population[i] = rand_whale - A * D
            else:
                distance_to_best = abs(best - population[i])
                b = 1
                l = (np.random.rand() * 2) - 1 
                population[i] = distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l) + best

        # Clip positions
        population = np.clip(population, -5.12, 5.12)

        # Evaluate fitness
        scores = np.array([rastrigin(ind) for ind in population])

        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best = population[current_best_idx].copy()
            best_score = scores[current_best_idx]

        history.append(population.copy())
        best_score_history.append(best_score)

    return np.array(history), best_score_history

# GOA Algorithm Implementation
class GOA_Params:
    def __init__(self, num_grasshoppers, dimensions, iterations, c1=0.5, c2=0.5, c3=0.1):
        self.num_grasshoppers = num_grasshoppers
        self.dimensions = dimensions
        self.iterations = iterations
        self.c1 = c1  # Attitude control parameter
        self.c2 = c2  # Gravity control parameter
        self.c3 = c3  # Wind control parameter

def grasshopper_optimization_algorithm(rastrigin, params):
    num_grasshoppers = params.num_grasshoppers
    dimensions = params.dimensions
    iterations = params.iterations
    c1 = params.c1
    c2 = params.c2
    c3 = params.c3

    # Initialize
    population = np.random.uniform(-5.12, 5.12, (num_grasshoppers, dimensions))
    scores = np.array([rastrigin(ind) for ind in population])

    # Find best
    best_idx = np.argmin(scores)
    best = population[best_idx].copy()
    best_score = scores[best_idx]

    history = [population.copy()]
    best_score_history = [best_score]

    for t in range(iterations):
        # Calculate distances and observation lines
        for i in range(num_grasshoppers):
            distance = population - population[i]
            distance_norm = np.linalg.norm(distance, axis=1)
            distance_norm[i] = 1e-8  # Prevent division by zero
            direction = distance / distance_norm[:, None]

            # Calculate gravity
            gravity = c1 * direction[np.argmin(scores)]

            # Gravity control
            gravity_control = c2 * gravity

            # Wind control
            wind_control = c3 * np.random.randn(dimensions)

            # Update position
            population[i] += gravity_control + wind_control

        # Clip positions
        population = np.clip(population, -5.12, 5.12)

        # Evaluate fitness
        scores = np.array([rastrigin(ind) for ind in population])

        # Update best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best = population[current_best_idx].copy()
            best_score = scores[current_best_idx]

        history.append(population.copy())
        best_score_history.append(best_score)

    return np.array(history), best_score_history

# Algorithm settings
num_iterations = 50
pso_params = PSO_Params(num_particles=30, dimensions=2, iterations=num_iterations)
de_params = DE_Params(pop_size=30, dimensions=2, iterations=num_iterations)
sa_params = SA_Params(initial_temp=100, final_temp=1, iterations=num_iterations, alpha=0.95)
woa_params = WOA_Params(num_whales=30, dimensions=2, iterations=num_iterations)
goa_params = GOA_Params(num_grasshoppers=30, dimensions=2, iterations=num_iterations)

# Execute algorithms and get histories
pso_history, pso_best_score_history = particle_swarm_optimization(rastrigin, pso_params)
de_history, de_best_score_history = differential_evolution(rastrigin, de_params)
sa_history, sa_best_score_history = simulated_annealing(rastrigin, sa_params)
woa_history, woa_best_score_history = whale_optimization_algorithm(rastrigin, woa_params)
goa_history, goa_best_score_history = grasshopper_optimization_algorithm(rastrigin, goa_params)

# Determine number of frames
min_frames = min(len(pso_history), len(de_history), len(sa_history),
                 len(woa_history), len(goa_history))

# Keep equal frames for all algorithms
pso_history = pso_history[:min_frames]
de_history = de_history[:min_frames]
sa_history = sa_history[:min_frames]
woa_history = woa_history[:min_frames]
goa_history = goa_history[:min_frames]

# Updated to improve the graphical display
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create surface plot for Rastrigin function
X = np.linspace(-5.12, 5.12, 400)
Y = np.linspace(-5.12, 5.12, 400)
X, Y = np.meshgrid(X, Y)
Z = rastrigin([X, Y])
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Add scatter plots with varied markers
ax.scatter(pso_history[0][:, 0], pso_history[0][:, 1], 
           [rastrigin(p) for p in pso_history[0]], 
           color='red', label='PSO', edgecolors='k', s=50)

ax.scatter(de_history[0][:, 0], de_history[0][:, 1], 
           [rastrigin(p) for p in de_history[0]], 
           color='blue', label='DE', edgecolors='k', s=50)

ax.scatter(sa_history[0][0], sa_history[0][1], 
           rastrigin(sa_history[0]), color='green', 
           label='SA', edgecolors='k', marker='X', s=100)

ax.scatter(woa_history[0][:, 0], woa_history[0][:, 1], 
           [rastrigin(p) for p in woa_history[0]], 
           color='orange', label='WOA', edgecolors='k', s=50)

ax.scatter(goa_history[0][:, 0], goa_history[0][:, 1], 
           [rastrigin(p) for p in goa_history[0]], 
           color='purple', label='GOA', edgecolors='k', s=50)

# Add global minimum marker
ax.scatter(0, 0, rastrigin([0, 0]), color='yellow', marker='*', s=200)

# Set labels and title
ax.set_title('Optimization Algorithms on Rastrigin Function (3D View)')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Function Value')

# Display legend
ax.legend(loc='upper right')

# Animation Update Function
# Animation Update Function
# Animation Update Function
def update(frame):
    # Clear the previous plot
    ax.cla()

    # Replot surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # Update positions for each algorithm
    pso_positions = pso_history[frame]
    de_positions = de_history[frame]
    sa_position = sa_history[frame]
    woa_positions = woa_history[frame]
    goa_positions = goa_history[frame]

    # Plot each algorithm's positions
    ax.scatter(pso_positions[:, 0], pso_positions[:, 1], 
               [rastrigin(p) for p in pso_positions], 
               color='red', edgecolors='k', s=50, 
               label=f"PSO: {pso_best_score_history[frame]:.4f}")

    ax.scatter(de_positions[:, 0], de_positions[:, 1], 
               [rastrigin(p) for p in de_positions], 
               color='blue', edgecolors='k', s=50, 
               label=f"DE: {de_best_score_history[frame]:.4f}")

    ax.scatter(sa_position[0], sa_position[1], 
               rastrigin(sa_position), color='green', 
               marker='X', edgecolors='k', s=100, 
               label=f"SA: {sa_best_score_history[frame]:.4f}")

    ax.scatter(woa_positions[:, 0], woa_positions[:, 1], 
               [rastrigin(p) for p in woa_positions], 
               color='orange', edgecolors='k', s=50, 
               label=f"WOA: {woa_best_score_history[frame]:.4f}")

    ax.scatter(goa_positions[:, 0], goa_positions[:, 1], 
               [rastrigin(p) for p in goa_positions], 
               color='purple', edgecolors='k', s=50, 
               label=f"GOA: {goa_best_score_history[frame]:.4f}")

    # Highlight the global minimum
    ax.scatter(0, 0, rastrigin([0, 0]), color='yellow', marker='*', s=200)

    # Update title with the current iteration
    ax.set_title(f"Optimization Algorithms on Rastrigin Function (Iteration {frame + 1})")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Function Value')

    # Add legend
    ax.legend(loc='upper right')

    return ax

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=min_frames, interval=200)

# Show the animation
plt.show()

# Print the final best results of each algorithm
print("Final best results of the algorithms:")
print(f"PSO: {pso_best_score_history[-1]}")
print(f"DE: {de_best_score_history[-1]}")
print(f"SA: {sa_best_score_history[-1]}")
print(f"WOA: {woa_best_score_history[-1]}")
print(f"GOA: {goa_best_score_history[-1]}")
