import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lotka-Volterra equations
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    prey, predator = z
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]

# Parameters for the Lotka-Volterra model
alpha = 0.1  # Prey birth rate
beta = 0.02  # Predation rate
delta = 0.01  # Predator reproduction rate
gamma = 0.1  # Predator death rate

# Initial conditions: 40 prey and 9 predators
initial_conditions = [40, 9]

# Time points where the solution is computed
t_span = (0, 200)  # Time range
t_eval = np.linspace(*t_span, 500)  # 500 time points

# Solve the ODEs
solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, delta, gamma), t_eval=t_eval)

# Create a DataFrame to hold the results
df = pd.DataFrame({
    'time': solution.t,
    'prey': solution.y[0],
    'predator': solution.y[1]
})

# Display the first few rows of the DataFrame
print(df.head())

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(df['time'], df['prey'], label='Prey')
plt.plot(df['time'], df['predator'], label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Lotka-Volterra Predator-Prey Model')
plt.legend()
plt.grid(True)
plt.show()