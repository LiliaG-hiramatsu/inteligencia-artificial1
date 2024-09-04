import random
import math

# Inicializar el tablero
tablero = ["-", "-", "-",
           "-", "-", "-",
           "-", "-", "-"]

ganador = None # Definimos una variable global que usaremos para identificar cuando alguien gana el juego

# Mostrar el tablero en pantalla
def mostrar_tablero():
    print("\n")
    print(tablero[0] + " | " + tablero[1] + " | " + tablero[2] + "      1 | 2 | 3")
    print(tablero[3] + " | " + tablero[4] + " | " + tablero[5] + "      4 | 5 | 6")
    print(tablero[6] + " | " + tablero[7] + " | " + tablero[8] + "      7 | 8 | 9")
    print("\n")

# Verificar si hay un ganador
def verificar_ganador(jugador):
    # Verificar filas, columnas y diagonales
    combinaciones_ganadoras = [(0, 1, 2), (3, 4, 5), (6, 7, 8), # Filas
                               (0, 3, 6), (1, 4, 7), (2, 5, 8), # Columnas
                               (0, 4, 8), (2, 4, 6)]            # Diagonales
    for comb in combinaciones_ganadoras: # Recorre cada combinación ganadora (tuplas)
        if tablero[comb[0]] == tablero[comb[1]] == tablero[comb[2]] == jugador: # Verifica si alguna de las tuplas tienen todas x ó todas o
            return True # Entonces si hay ganador
    return False # No hay ganador

# Realizar una jugada del jugador humano
def jugar_humano(jugador):
    jugada_valida = False # Esta variable es para asegurarnos de que el jugador humano seleccione una casilla válida
    while not jugada_valida: # El bucle se ejecuta hasta que jugada_valida sea True
        posicion = int(input(f"Jugador {jugador}, elige una posición (1-9): ")) - 1 # Resta 1 porque las posiciones van desde el 0
        if tablero[posicion] == "-": # Si la casilla está vacóia entonces es válido
            tablero[posicion] = jugador # Coloca la x en esa posición
            jugada_valida = True # Cuando es True sale del bucle
        else:
            print("Posición ocupada, elige otra.")
    mostrar_tablero() # Muestra como queda el tablero luego de la jugada

# Realizar una jugada de la IA utilizando recocido simulado
def jugar_ia(jugador):
    temperatura = 1.0
    tasa_enfriamiento = 0.9
    mejor_movimiento = recocido_simulado(jugador, temperatura, tasa_enfriamiento)
    tablero[mejor_movimiento] = jugador
    mostrar_tablero()

# Algoritmo de recocido simulado para elegir el mejor movimiento
def recocido_simulado(jugador, temperatura, tasa_enfriamiento):
    mejor_movimiento = None
    while temperatura > 0.1:
        # Obtener las casillas vacías
        casillas_vacias = []
        for i in range(9):
            if tablero[i] == "-":
                casillas_vacias.append(i)

        if not casillas_vacias:
            break
        
        # Escoger una casilla aleatoria
        movimiento = random.choice(casillas_vacias)
        
        # Simular el movimiento
        tablero[movimiento] = jugador
        if verificar_ganador(jugador):
            tablero[movimiento] = "-"
            return movimiento  # Si este movimiento gana, lo retorna
        
        # Cálculo de energía (evaluar cuán bueno es el movimiento)
        energia_actual = evaluar_energia(jugador)
        
        # Deshacer el movimiento simulado
        tablero[movimiento] = "-"
        
        # Verificar si el movimiento mejora o aceptar por probabilidad
        if mejor_movimiento is None or energia_actual > evaluar_energia(jugador):
            mejor_movimiento = movimiento
        elif random.random() < math.exp(-abs(energia_actual) / temperatura):
            mejor_movimiento = movimiento
        
        # Enfriar la temperatura
        temperatura *= tasa_enfriamiento
    
    return mejor_movimiento

# Evaluar la energía del tablero para el jugador (entre más cercano a ganar, mejor)
def evaluar_energia(jugador):
    # La energía es alta si hay muchas líneas con posibilidades de ganar
    energia = 0
    combinaciones_ganadoras = [(0, 1, 2), (3, 4, 5), (6, 7, 8), # Filas
                               (0, 3, 6), (1, 4, 7), (2, 5, 8), # Columnas
                               (0, 4, 8), (2, 4, 6)]            # Diagonales
    for comb in combinaciones_ganadoras:
        valores = [tablero[comb[0]], tablero[comb[1]], tablero[comb[2]]]
        if valores.count(jugador) == 2 and valores.count("-") == 1: # De las combinaciones ganadoras posibles, cuenta si hay 2 casillas marcadas con "o" y 1 casilla vacía (gran chance de ganar)
            energia += 10  # Alta energía para casi ganar
        elif valores.count(jugador) == 1 and valores.count("-") == 2: # De las combinaciones ganadoras posibles, cuenta si hay sólo una casilla con la marca "o" y 2 casillas vacías (la chance de ganar es baja)
            energia += 1  # Suma algo de energía porque la posibilidad todavía existe
    return energia

# Función principal del juego
def jugar():
    global ganador
    print("Empieza el juego!")
    mostrar_tablero()
    
    for i in range(5): # Como el jugador 1 empieza, tiene hasta 5 posibilidades de jugar
        if ganador is None:
            # Turno del jugador 1 (humano)
            jugar_humano("x")
            if verificar_ganador("x"):
                print("¡Felicidades! Jugador 1 gana el juego.")
                ganador = "x"
                break
            
            # Turno del jugador 2 (IA)
            if i < 4:  # La IA juega hasta 4 turnos
                jugar_ia("o")
                if verificar_ganador("o"):
                    print("¡Felicidades! La IA gana el juego.")
                    ganador = "o"
                    break
        else:
            print("Empate!")
            break

# Iniciar el juego
jugar()