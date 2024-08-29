### Ejercicio 4 TP3
import math
import random

def f(x):
    return (math.sin(x))/(x+0.1)

def hill_climbing(x_inicial, paso, error):
    
    x_actual = x_inicial
    f_actual = f(x_actual)

    while True:
        if f(x_actual+paso) > f_actual:
            x_vecino = x_actual + paso
        else:
            x_vecino = x_actual - paso
        
        f_vecino = f(x_vecino)
        
        if f_vecino > f_actual:
            x_actual = x_vecino
            f_actual = f_vecino
        else:
            if abs(f_vecino - f_actual) < error:
                break
            
    return x_actual, f_actual

x_inicial = random.uniform(-10, -6)
paso = 0.01
error = 0.1

x_max, f_max = hill_climbing(x_inicial, paso, error)

print(f'El valor maximo de la funcion es: {round(f_max, 2)} para x igual a: {round(x_max, 2)}')
