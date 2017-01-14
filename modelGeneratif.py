#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import numpy as np, numpy.random
from numpy.random import choice
from pprint import pprint

# Generation des causes du modèle
# params : N le nombre de causes totales
# return : Un tableau de taille N contenant les probabilités d'apparition de chaque cause
def genCauses(N):

	# c le tableau des probas des causes
	c = []

	maxProba = 1

	# On remplit le tableau avec des probabilités sommant à 1
	for i in range(N-1):
		c.append(random.uniform(0,maxProba))
		maxProba = maxProba - c[i]
	c.append(maxProba)

	# On retourne le tableau de probabilités
	return c

# Methode similaire à celle ci-dessus. Cependant, les valeurs sont générées selon
# une loi normale centrée en 0.5 réduite pour obtenir des valeurs entre 0 et 1.
# On divise ensuite les valeurs obtenue par leur somme pour qu'elles somment à 1
# Cette methode sert à generer des probas sans valeurs extrèmes.
# params : N le nombre de causes totales
# return : Un tableau de taille N contenant les probabilités d'apparition de chaque cause
def genCausesGaussienne(X):
	
	# On choisit des valeurs selon une loi normale centree reduite
	c = np.random.normal(0.5,0.25,X)
	# Normalisation des valeurs
	cReturn = [c[x] / sum(c) for x in range(len(c))]

	# On retourne le tableau de probabilités
	return cReturn

# Generation des probabilités d'observation conditionelles aux causes
# params : C le tableau des causes
#		   N le nombre d'éléments du contexte
# return : Un tableau de taille C composé de vecteurs représentant les probabilités
#		   de chaque élement conditionellement au c actuel
def genProbasObservations(C,N):

	# Probas le tableau de probabilités p(f | c)
	probas = []

	# Pour chaque cause existante, on remplit un tableau de probabilités conditionelles
	for c in range(len(C)):
		# tab le tableau de probas conditionnel à c
		# tab[0] = [p(non US|c), p(US|c)]
		# tab[1] = [p(non CS|c), p(CS|c)]
		# tab[2] = [p(contexteA|c), p(contexteB|c), ...] contenant N valeurs
		tab = []
		# tab[0]
		rand = random.random()
		tab.append([rand,1-rand])
		# tab[1]
		rand = random.random()
		tab.append([rand,1-rand])
		# tab[2]. Le remplissage se fait de la même manière que pour générer le tableau
		# de probas des causes. On utilise donc la même methode, bien que les deux tableaux
		# correspondent à des choses differentes
		tab.append(genCauses(N))

		probas.append(tab)

	return probas

# Generation des observations à partir de probabilités de causes et de probabilités
# conditionnelles d'observations données
# params : C le tableau des causes
#		   F le tableau des probas conditionelles
#		   N le nombre d'observations désirées
# return : Un tableau de taille N contenant pour chaque case un vecteur d'observations
def genObservations(C,F,N):

	# Tableau contenant toutes les observations
	observations = []

	# On genere N observations
	for i in range(N):
		# On choisit l'une des causes à partir du tableau de probas des causes
		# (On prend l'index d'une valeur tirée dans le tableau C avec les probas du tableau)
		cause = choice(range(len(C)),1,p=C)

		# Tableau des valeurs d'observations
		tab = []
		# Valeur de US (0 indique son absence, 1 sa présence)
		# On tire une valeur du tableau [0, 1] avec les probas de f conditionnelles à la cause choisie
		us = choice(2,1,p=F[int(cause)][0])[0]
		tab.append(us)
		# Valeur de CS (0 indique son absence, 1 sa présence)
		# Tirage similaire à US
		cs = choice(2,1,p=F[int(cause)][1])[0]
		tab.append(cs)
		# Valeur du contexte choisi
		# Tirage similaire à précedemment
		contexte = choice(range(len(F[int(cause)][2])),1,p=F[int(cause)][2])[0]
		tab.append(contexte)

		# On ajoute l'observation effectuée au tableau des observations
		observations.append(tab)

		print "( Cause : {}".format(cause)

	# On renvoie l'ensemble des observations 
	return observations



tabCauses = genCausesGaussienne(5)
print "Probas Causes:\n"
pprint(tabCauses)
print "\n"

tabProbas = genProbasObservations(tabCauses,3)
print "Probas Condtionnelles Observations:\n"
pprint(tabProbas)
print "\n"

observations = genObservations(tabCauses,tabProbas,10)
print "Observations:\n"
pprint(observations)
print "\n"