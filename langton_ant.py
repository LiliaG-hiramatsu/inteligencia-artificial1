import pygame as pg  # Importo la librería Pygame para crear la ventana y dibujar la hormiga
from collections import deque  # Importo 'deque' que es una cola doblemente enlazada, me permite agregar y quitar elementos de ambos extremos.
# La usamos para manejar de forma eficiente las rotaciones de la hormiga ya que tiene el método 'rotate' que mueve los elementos de la colección
# hacia la derecha o izquierda por un número determinado de posiciones

class Ant:
    def __init__(self, app, pos, color):
        # Inicializo la hormiga con la aplicación, la posición inicial y el color
        self.app = app  # Referencia a la aplicación principal
        self.color = color  # Color de la hormiga
        self.x, self.y = pos  # Coordenadas iniciales de la hormiga
        self.increments = deque([(1, 0), (0, 1), (-1, 0), (0, -1)])  # Cola que maneja las direcciones (derecha, abajo, izquierda, arriba)
        self.history = deque(maxlen=1000)  # Guardo los últimos 1000 estados para detectar patrones repetitivos
        self.avenue_found = False  # Indica si la hormiga ya encontró la avenida

    def run(self):
        # Guardo el estado actual (posición y dirección) de la hormiga en la historia
        state = (self.x, self.y, tuple(self.increments))
        self.history.append(state)  # Agrego el estado actual al historial
        
        # Si aún no se ha encontrado la avenida, busco si el patrón actual se repite
        if not self.avenue_found and self.history.count(state) > 10:  # Si el estado actual aparece más de 10 veces, considero que encontró la avenida
            self.app.iterations_until_avenue = self.app.iterations  # Registro cuántas iteraciones tomó encontrar la avenida
            self.avenue_found = True  # Marco que se encontró la avenida
            print(f"La hormiga ha encontrado la avenida después de {self.app.iterations_until_avenue} iteraciones.")
        
        # Manejo el cambio de color de la celda y el movimiento de la hormiga
        value = self.app.grid[self.y][self.x]  # Obtengo el valor de la celda actual (True o False)
        self.app.grid[self.y][self.x] = not value  # Cambio el color de la celda (de blanco a negro o viceversa)
        
        # Dibujo la celda con el nuevo color
        SIZE = self.app.CELL_SIZE
        rect = self.x * SIZE, self.y * SIZE, SIZE - 1, SIZE - 1
        if value:
            pg.draw.rect(self.app.screen, pg.Color('white'), rect)  # Si era blanco, ahora será negro
        else:
            pg.draw.rect(self.app.screen, self.color, rect)  # Si era negro, ahora será blanco
        
        # Rotación de la hormiga según el color de la celda (blanco: derecha, negro: izquierda)
        self.increments.rotate(1) if value else self.increments.rotate(-1)
        dx, dy = self.increments[0]  # Obtengo la nueva dirección de la hormiga
        self.x = (self.x + dx) % self.app.COLS  # Actualizo la posición de la hormiga en X (asegurando que no salga de la cuadrícula)
        self.y = (self.y + dy) % self.app.ROWS  # Actualizo la posición de la hormiga en Y (asegurando que no salga de la cuadrícula)

        # Si la hormiga choca con el borde, detengo el programa
        if self.x == 0 or self.x == self.app.COLS - 1 or self.y == 0 or self.y == self.app.ROWS - 1:
            pg.quit()  # Cierro Pygame
            exit()  # Salgo del programa

class App:
    def __init__(self, WIDTH=900, HEIGHT=500, CELL_SIZE=4):
        # Inicializo la aplicación Pygame
        pg.init()
        self.screen = pg.display.set_mode([WIDTH, HEIGHT])  # Configuro la ventana de Pygame
        self.clock = pg.time.Clock()  # Creo un reloj para controlar la velocidad de la simulación

        self.CELL_SIZE = CELL_SIZE  # Tamaño de cada celda
        self.ROWS, self.COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE  # Número de filas y columnas basado en el tamaño de las celdas
        self.grid = [[0 for col in range(self.COLS)] for row in range(self.ROWS)]  # Creo la cuadrícula donde se moverá la hormiga

        # Creo una instancia de la hormiga en el centro de la cuadrícula
        self.ant = Ant(app=self, pos=[self.COLS // 2, self.ROWS // 2], color=pg.Color('black'))
        self.iterations = 0  # Inicializo el contador de iteraciones totales
        self.iterations_until_avenue = 0  # Inicializo el contador de iteraciones hasta encontrar la avenida
    
    def run(self):
        # Bucle principal de la aplicación
        while True:
            self.ant.run()  # Ejecuto la lógica de la hormiga en cada iteración
            self.iterations += 1  # Incremento el contador de iteraciones

            [exit() for i in pg.event.get() if i.type == pg.QUIT]  # Permito salir del programa si se cierra la ventana
            pg.display.flip()  # Actualizo la pantalla para mostrar los cambios
            self.clock.tick()  # Para mantener la simulación a 60 FPS (frames por segundo) poner 60 como parámetro

if __name__ == '__main__':
    app = App()  # Creo una instancia de la aplicación
    app.run()  # Inicio la simulación
