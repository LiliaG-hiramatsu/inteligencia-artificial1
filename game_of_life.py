import time
import os
os.system("")

Cw=20
Ch=20
Lienzo=[' ' for _ in range((Cw*2+1)*Ch)] #Creamos el array de caracteres sobre el que vamos a pintar
for i in range(Ch):
	Lienzo[Cw*2*(i+1)]='\n'

Lienzo_base=Lienzo.copy() #Una copia a la que vamos a volver tras cada frame

def pintar_pixel(x,y,color_p): #pinta los pixeles del lienzo usando codigos de escape ANSI
	global Lienzo
	color = color_p*7+40 #Si es 1 es blanco, si es 0 es negro
	pos = y*Cw*2+x*2
	Lienzo[pos]=Lienzo[pos+1]="\x1b["+str(color)+"m \x1b[0m"

celulas_base = [[0 for i in range(Cw*2)] for j in range(Ch)]
celulas=celulas_base.copy()
live = []
celulas[5][7:10]=[1,1,1] #3 celulas iniciales para probar

def step(live):
	global celulas
	live.clear()
	
	#recorremos todas las celulas, si una celula en el siguiente frame debería estar viva agregamos sus coordenadas a la lista live
	for x in range(Cw):
		for y in range(Ch):
			celula = celulas[y][x]
			n = get_neighbors(x,y)
			if celula==1:
				if n==2 or n==3:
					live.append([x,y]) #si la celula está viva y tiene 2 o 3 vecinos vivos sigue viva
			elif n==3:
				live.append([x,y]) #si una celula muerta tiene 3 vecinos vivos cobra vida
	#celulas=celulas_base.copy() #borramos los datos del frame previo
	for j in range(len(celulas)):
		for i in range(len(celulas[j])):
			celulas[j][i]=0
	
	for cell in live:
		celulas[cell[1]][cell[0]]=1 #en todas las celulas que deberían estar vivas les escribimos 1


def get_neighbors(x, y): #Obtiene el número de vecinos vivos rodeando a una celula
	n = 0
	for i in range(x-1,x+2):
		for j in range(y-1,y+2):
			if not(i==x and j==y) and (Cw>i>=0 and Ch>j>=0):
				if celulas[j][i]==1: n+=1  #;print(str(i)+" "+str(j))
	return n

def renderizar():  #pinta todas las celulas vivas, convierte el Lienzo en un string y lo imprime 
	global Lienzo
	
	for cell in live:
		pintar_pixel(cell[0],cell[1],1)
	rend="".join(Lienzo)
	print(rend)

for i in range(10):
	step(live)
	renderizar()
	Lienzo=Lienzo_base.copy() #limpiamos el Lienzo después de cada frame.
	time.sleep(0.5)