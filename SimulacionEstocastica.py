import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
from pytwalk import pytwalk
from numpy import linspace
from scipy.integrate import odeint

# Ecuacion diferencial del modelo logistico
def logistic_differential_equation(x,t,K,L):
    return L * x * (K - x)

# Resolver la ecuación diferencial utilizando Odeint
def solve_logistic_differential_equation(t_data, K, L):
    x_values = odeint(logistic_differential_equation,[100],t_data,args=(K,L)).flatten()
    return np.array(x_values)

# Soporte para que los parametros sean positivos
def logistic_supp(params):
    return all(param > 0 for param in params)

# Función objetivo 
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

# Parametros reales para simluar datos
t_data = np.linspace(0, 10, 26)  # Tiempo
K_obs = 1000  # Valor verdadero de K
L_obs = 1/1000  # Valor verdadero de L
noise_std = 30  # Desviación estándar del ruido gaussiano

# Generar datos sintéticos
x_data = (1000/9)*np.exp(t_data)/(1+(1/9)*np.exp(t_data)) + np.random.normal(0, noise_std, size=len(t_data))
x_data[0]=100

# Parametros de la distribución a priori gamma
alpha_prior = 2
beta_prior = 0.001

# Dos valores iniciales
initial_params_chain1 = np.array([np.random.uniform(0, 1500), np.random.uniform(0, 0.002), np.random.uniform(0, 50)])
initial_params_chain2 = np.array([np.random.uniform(0, 1500), np.random.uniform(0, 0.002), np.random.uniform(0, 50)])

# Creamos el objeto Pytwalk
BayesLog = pytwalk(n=3, U=logistic_potential_energy, Supp=logistic_supp)
BayesLog.par_names = [r"$K$", r"$L$", r"$\sigma$"]

# Correr MCMC con diferentes valores iniciales
BayesLog.Run(T=100000, x0=initial_params_chain1, xp0=initial_params_chain2)

# Valores de la a posteriori de los parámetros
posterior_samples = BayesLog.Output


# Graficar histogramas de las muestras de parámetros aceptadas con curvas a priori
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

param_names = [r"$K$", r"$L$", r"$\sigma$"]
true_values = [1000, 0.001, 30]  # Valores verdaderos de K, L y sigma
burn_in = 1000

# Funciones de distribución a priori
prior_distributions = [
    lambda x: gamma.pdf(x, alpha_prior, scale=1/beta_prior),
    lambda x: gamma.pdf(x, alpha_prior, scale=1/beta_prior),
    lambda x: gamma.pdf(x, alpha_prior, scale=1/beta_prior)
]

x_range = [np.linspace(960, 1020, 500), np.linspace(0.0009, 0.00110, 500), np.linspace(10, 60, 500)]

for i, ax in enumerate(axes):
    ax.hist(BayesLog.Output[burn_in:, i], bins=30, density=True, alpha=0.5, color='blue', label='Distribución a posteriori')
    ax.axvline(true_values[i], color='red', linestyle='dashed', linewidth=1, label=f'Valor verdadero = {true_values[i]}')
    ax.plot(x_range[i], prior_distributions[i](x_range[i]), 'k',color='m', linewidth=2, label='Distribución a priori')
    ax.set_title(f'Histograma de {param_names[i]}')
    ax.set_xlabel('Valores')  
    ax.set_ylabel('Densidad')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()


# Obtener muestras de los parámetros
K_samples = posterior_samples[:, 0]
L_samples = posterior_samples[:, 1]
sigma_samples = posterior_samples[:, 2]

#Calculamos los percentiles 5 y 95
pK5 = np.percentile(K_samples, 5)
pK95 = np.percentile(K_samples, 95)

pL5 = np.percentile(L_samples, 5)
pL95 = np.percentile(L_samples, 95)

# Filtrar el vector entre los percentiles 5 y 95
K_samples_filtrado = [x for x in K_samples if pK5 <= x <= pK95]
K_samples=np.array(K_samples_filtrado)
L_samples_filtrado = [x for x in L_samples if pL5 <= x <= pL95]
L_samples=np.array(L_samples_filtrado)


#Se tomara una muestra de tamaño 5000  de la distribucion a posteriri
Aran= np.random.randint(low=0,high= np.min([len(K_samples),len(L_samples)]),size= 5000)

t1=linspace(0, 10, num=3000) #The sample size is 30, a grid of size 30


# Curvas del modelo con los parámetros de la a posteriori
#Area sombreada 
plt.figure(figsize=(10, 6))
for i in range(len(t1)):
    K_i = K_samples[Aran[i]]
    L_i = L_samples[Aran[i]]
    x_values = solve_logistic_differential_equation(t_data, K_i, L_i)
    plt.plot(t_data, x_values, color='gray')


# Configuración de la gráfica
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.plot(t_data, x_data, 'ro', label='Datos Observados')
plt.plot(t_data, solve_logistic_differential_equation(t_data, np.median(K_samples), np.median(L_samples)),
         color='r', label="Mejor ajuste")
#plt.plot(t_data, solve_logistic_differential_equation(t_data, K_obs, L_obs), color='b', label="Real")
plt.plot(t_data, (1000/9)*np.exp(t_data)/(1+(1/9)*np.exp(t_data)), color='b', label="Real")
plt.title('Curvas del modelo logístico utilizando valores de la a posteriori')
plt.legend()
plt.grid(True)
plt.show()

