#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from modelGeneratif import *
import random
from pprint import pprint
from numpy.random import choice

# Fonction de particle filter
# params : Observations le tableau des observations à un temps t
#		   alpha le paramètre de concentration
#		   M le nombre de particules à générer
# return : je sais pas trop
def particleFilter(observations, alpha = 1, M = 200):


	# Tableau de taille M dans lequel, pour chaque particule,
	# on stock la liste des causes associées à chaque observations
	causesParticles = []

	# Pour chaque particule, la premiere observation est associée à
	# la cause 1
	for m in range(M):
		causesParticles.append([0])

	# On parcours toutes les observations après la 1ere pas à pas
	for t in range(1,len(observations)):

		# Pour chaque particule
		for m in range(M):
			# Le nombre de causes déterminées
			nbCauses = max(causesParticles[m]) + 1

			# Tableau de probabilité de chaque cause pour la particule m actuelle
			p = [0 for _ in range(nbCauses + 1)]

			# Equation (1)
			eq1 = [0 for _ in range(nbCauses + 1)]
			# Pour chaque causes connue, P(k) = Nk / t-1+alpha
			for k in range(len(p) - 1):
				eq1[k] = causesParticles[m].count(k) / (t + alpha)
			# Proba d'obtenir une nouvelle cause
			eq1[len(eq1)-1] = alpha / (t + alpha)

			# Equation (4)
			eq4 = [0 for _ in range(nbCauses + 1)]
			# Pour chaque cause connue k
			for k in range(len(p)):
				# product = produit des probas
				product = 1
				# Pour chaque feature observé, on calcule sa proba selon k
				for i in range(len(observations[0])):
					# La somme du nombre d'occurences
					somme = sum(((observations[_t][i] == observations[t][i]) and (causesParticles[m][_t] == k)) for _t in range(t))
					somme += 1

					# On divise par la somme du nombre de cas + le nombre de j obtenus
					maxJ = max([observations[_t][i] for _t in range(t)]) + 1
					somme /= (causesParticles[m].count(k) + maxJ)

					product *= somme


				eq4[k] = product

			# Equation (3)
			# Le denominateur est égal à la somme sur tous les choix possibles du produit des probas
			denominateur = sum(eq1[k] * eq4[k] for k in range(len(p)))

			# On calcule les probas finales en multipliant les 2 equations du numerateur EQ1 et EQ4
			# et en divisant le produit par le denominateur.
			for k in range(len(p)):
				p[k] = eq1[k] * eq4[k] / denominateur

			pprint(p)

			# On tire la nouvelle cause de l'observation selon la distribution de probas calculée
			causesParticles[m].append(choice(len(p),1,p=p)[0])



		print("-- NEW TRIAL --")

tabCauses = genCausesGaussienne(5)
tabProbas = genProbasObservations(tabCauses,2)
observations,causes = genObservations(tabCauses,tabProbas,20)

particleFilter(observations)

