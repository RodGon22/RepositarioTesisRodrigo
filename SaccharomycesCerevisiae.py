import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
from numpy import linspace
from pytwalk import pytwalk
from scipy.integrate import odeint


# Ecuación diferencial del modelo logístico
def logistic_differential_equation(x,t, K, L):
    return L * x * (K - x)

# Resolver la ecuación diferencial utilizando el método de Euler
def solve_logistic_differential_equation(t_data, K, L):
    x_values = odeint(logistic_differential_equation,[0.2],t_data,args=(K,L)).flatten()
    return np.array(x_values)

# Soporte para que los parámetros sean positivos
def logistic_supp(params):
    return all(param > 0 for param in params)

# Energía para el modelo logístico
def logistic_potential_energy(params):
    if not logistic_supp(params[:-1]):  # Excluir sigma del chequeo de soporte
        return -np.inf  # Devolver menos infinito si los parámetros no están en el soporte
    log_prior_K = gamma.logpdf(params[0], alpha_prior, scale=1/beta_prior)
    log_prior_L = gamma.logpdf(params[1], alpha_prior, scale=1/beta_prior)
    log_prior_sigma = gamma.logpdf(params[2], alpha_prior, scale=1/beta_prior)
    return -log_likelihood(params[:-1], t_data, x_data, params[-1]) - log_prior_K - log_prior_L - log_prior_sigma

def log_likelihood(params, t_data, x_data, sigma):
    K, L = params
    x_model = solve_logistic_differential_equation(t_data, K, L)
    log_likelihood = np.sum(norm.logpdf(x_data, loc=x_model, scale=sigma))
    return log_likelihood


#Datos de la tabla de la poblacion
x_data=np.array([.2,.33,.5,1.1,1.4,3.1,3.5,9,10,25.4,27,55,76,115,160,162,190,193,190,209,190,210,200,215,220,200,180,213,210,210,220,213,200,211,200,208,230])
std_datos=np.std(x_data)

#Grafica de los datos de la tabla
plt.figure(figsize=(10, 6))
t_data = np.arange(len(x_data))
plt.plot(t_data,x_data,'.r')
plt.ylabel('Tamaño de la poblacion $x10^6/ml$')
plt.xlabel('Tiempo (h)')
plt.show()

# Grafica de minimos cuadrados
plt.figure(figsize=(10, 6))
plt.xlabel('Tiempo (hrs)')
plt.ylabel('Tamaño de la poblacion $x10^6/ml$')
plt.plot(t_data, x_data, 'bo', label='Datos Observados')
plt.plot(t_data, solve_logistic_differential_equation(t_data, 208.85552246, 0.54975787/208.85552246), color='r', label="Aproximada por mínimos cuadrados")
plt.title('(Saccharomyces cerevisiae).')
plt.legend()
plt.grid(True)
plt.show()


# Parámetros de la distribución a priori gamma
alpha_prior = 2
beta_prior = 0.01

#Funciones de distribución a priori
prior_distributions = [
    lambda x: gamma.pdf(x, alpha_prior, scale=1/beta_prior),
    lambda x: gamma.pdf(x, alpha_prior, scale=1/beta_prior),
    lambda x: gamma.pdf(x, alpha_prior, scale=1/beta_prior)
]

# Parámetros iniciales
initial_params_chain1 = np.array([np.random.uniform(0, 300), np.random.uniform(0, 0.002), np.random.uniform(0, std_datos)])
initial_params_chain2 = np.array([np.random.uniform(0, 300), np.random.uniform(0, 0.002), np.random.uniform(0, std_datos)])

# Configurar el objeto pytwalk
Cerve = pytwalk(n=3, U=logistic_potential_energy, Supp=logistic_supp)
Cerve.par_names = [r"$K$", r"$L$", r"$\sigma$"]

# Realizamos MCMC con diferentes valores iniciales
Cerve.Run(T=100000, x0=initial_params_chain1, xp0=initial_params_chain2)

# Valores de la a posteriori de los parámetros
posterior_samples = Cerve.Output

# Obtener muestras de los parámetros
K_samples = posterior_samples[:, 0]
L_samples = posterior_samples[:, 1]
sigma_samples = posterior_samples[:, 2]

pK5 = np.percentile(K_samples, 5)
pK95 = np.percentile(K_samples, 95)

pL5 = np.percentile(L_samples, 5)
pL95 = np.percentile(L_samples, 95)

# Filtrar  entre los percentiles 5 y 95
K_samples_filtrado = [x for x in K_samples if pK5 <= x <= pK95]
K_samples=np.array(K_samples_filtrado)
L_samples_filtrado = [x for x in L_samples if pL5 <= x <= pL95]
L_samples=np.array(L_samples_filtrado)

# Graficar histogramas de las muestras de parámetros aceptadas con histogramas a priori
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

param_names = [r"$K$", r"$L$", r"$\sigma$"]
true_values = [208.85552246, 0.00263224, np.mean(sigma_samples)]  # Valores estimados por Vazquez
true_values2 = [211.538461, 0.0026, np.mean(sigma_samples)]  # Valores estimados por Stewart y Day

burn_in = 1000
#Limites de la distribucion a priori
x_range = [np.linspace(200, 219, 500), np.linspace(0.0024, 0.00285, 500), np.linspace(5, 16, 500)]

for i, ax in enumerate(axes):
    ax.hist(Cerve.Output[burn_in:, i], bins=50, density=True, alpha=0.5, color='blue', label='Distribución a posteriori')
    ax.axvline(true_values[i], color='y', linestyle='solid', linewidth=1, label=f'Valor estimado Vázquez = {true_values[i]}')
    ax.axvline(true_values[i], color='k', linestyle='dashed', linewidth=1, label=f'Valor estimado Stewart y Day = {true_values2[i]}')
    ax.plot(x_range[i], prior_distributions[i](x_range[i]), 'k',color='m', linewidth=2, label='Distribución a priori')
    ax.set_title(f'Histograma de {param_names[i]}')
    ax.set_xlabel('Valores')
    ax.set_ylabel('Densidad')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

#Se tomara una muestra de tamaño 5000  de la distribucion a posteriri
Aran= np.random.randint(low=0,high= np.min([len(K_samples),len(L_samples)]),size= 5000)

t1=linspace(0, 10, num=3000) #The sample size is 30, a grid of size 30


# Graficar curvas del modelo con los parámetros de la posteriori 
plt.figure(figsize=(10, 6))
for i in range(len(t1)):
    K_i = K_samples[Aran[i]]
    L_i = L_samples[Aran[i]]
    x_values = solve_logistic_differential_equation(t_data, K_i, L_i)
    plt.plot(t_data, x_values, color='gray')

# Grafica del area sombreada, incertidumbre
plt.xlabel('Tiempo (hrs)')
plt.ylabel('Tamaño de la poblacion $x10^6/ml$')
plt.plot(t_data, x_data, 'bo', label='Datos Observados')
plt.plot(t_data, solve_logistic_differential_equation(t_data, np.mean(K_samples), np.mean(L_samples)),
         color='b', label="Mejor ajuste")
plt.plot(t_data, solve_logistic_differential_equation(t_data, 211.538461, 0.0026),linestyle='dashed', color='k', label="Ajuste por Vazquez")
plt.plot(t_data, solve_logistic_differential_equation(t_data, 208.85552246, 0.54975787/208.85552246),linestyle='dashed', color='y', label="Ajuste de Stewart y Day")
plt.title('Curvas del modelo logístico utilizando valores de la a posteriori')
plt.legend()
plt.grid(True)
plt.show()
