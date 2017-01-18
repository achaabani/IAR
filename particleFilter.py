#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from modelGeneratif import *
import random
from pprint import pprint
from numpy.random import choice

import math

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


	# Tableau P contenant les probabilités d'avoir une cause donnée à partir 
	# des observations précédentes. Calculé à partir des causes trouvérs pour les particules
	P = []
	P.append([1])

	# Causes estimées à chaque pas de temps, tirées grace aux 
	# distributions de probas contenues dans P
	causes = []
	causes.append(0)

	# Prediction du renforcement ou non (0 pour non, 1 pour oui)


	# On parcours toutes les observations après la 1ere pas à pas
	for t in range(1,len(observations)):

		allEq3 = []

		# Pour chaque particule on calcule p(cl = k | cl1:t-1, F1:t)
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
			eq3 = [0 for _ in range(nbCauses + 1)]
			# Le denominateur est égal à la somme sur tous les choix possibles du produit des probas
			denominateur = sum(eq1[k] * eq4[k] for k in range(len(p)))

			# On calcule les probas finales en multipliant les 2 equations du numerateur EQ1 et EQ4
			# et en divisant le produit par le denominateur.
			for k in range(len(p)):
				eq3[k] = eq1[k] * eq4[k] / denominateur

			# On ajoute au tableau de probas de chauqe particule les probas pour la particule actuelle
			allEq3.append(eq3)

		# Comme toutes les particules n'ont pas forcement le meme nombre de causes
		# On ajoute les probas = 0 pour les causes non connues pour chaque particule
		maxlen = max(len(x) for x in allEq3)

		for x in allEq3:
			if(len(x) != maxlen):
				x.extend([0 for _ in range(maxlen-len(x))])

		# On calcule la nouvelle distribution de probas pour le tirage des particules

		# p le tableau de probas de p(cl = k)
		pl = [0 for _ in range(maxlen)]
		
		for k in range(maxlen):
			pl[k] = sum(allEq3[l][k] for l in range(M)) / M

		# On resample des nouvelles particules (l) selon les probas inconditionneles p(cl = k)
		for m in range(M):
			# On tire la nouvelle cause de l'observation selon la distribution de probas calculée
			causesParticles[m].append(choice(len(pl),1,p=pl)[0])			

		# Maintenant que les causes pour chaque particules sont tirées, on cherche
		# desormais les probas à partir des causes générées dans chaque particule

		# Pour chaque cause possible, on compte le nombre de fois où elle apparait
		# pour une temps t donné, puis on normalise
		tab = [0 for _ in range(maxlen)]
		for k in range(maxlen):
			tab[k] = sum((causesParticles[m][t] == k) for m in range(M)) / M
		P.append(tab)
		causes.append(choice(maxlen,1,p=P[-1])[0])

		# //////////////////////////////////////// #

		# Calcul de la probabilité d'avoir renforcement selon les observations précédentes

		# Vt = P(f_t,1 = US | f_t,2:D, F_1:t-1)

		Vt = [0, 0]

		for m in range(M):

			# Le nombre de causes déterminées
			nbCauses = max(causesParticles[m]) + 1

			# Tableau de probabilité de chaque cause pour la particule m actuelle
			p = [0 for _ in range(nbCauses + 1)]

			# Equation (1)
			eq1bis = [0 for _ in range(nbCauses + 1)]
			# Pour chaque cause tirée dans le tableau des "vraies" causes choisies (pas celles de particules)
			# P(k) = Nk / t-1+alpha
			for k in range(len(p) - 1):
				eq1bis[k] = causes.count(k) / (t + alpha)
			# Proba d'obtenir une nouvelle cause
			eq1bis[len(eq1bis)-1] = alpha / (t + alpha)

			# Equation (4)
			eq4bis = [0 for _ in range(nbCauses + 1)]
			# Pour chaque cause connue k
			for k in range(len(p)):
				# product = produit des probas
				product = 1
				# Pour chaque feature observé, sauf le 1er, on calcule sa proba selon k
				for i in range(1,len(observations[0])):
					# La somme du nombre d'occurences
					somme = sum(((observations[_t][i] == observations[t][i]) and (causesParticles[m][_t] == k)) for _t in range(t))
					somme += 1

					# On divise par la somme du nombre de cas + le nombre de j obtenus
					maxJ = max([observations[_t][i] for _t in range(t)]) + 1
					somme /= (causesParticles[m].count(k) + maxJ)

					product *= somme


				eq4bis[k] = product

			# Equation (3)
			eq3bis = [0 for _ in range(nbCauses + 1)]
			# Le denominateur est égal à la somme sur tous les choix possibles du produit des probas
			denominateur = sum(eq1bis[k] * eq4bis[k] for k in range(len(p)))

			# On calcule les probas finales en multipliant les 2 equations du numerateur EQ1 et EQ4
			# et en divisant le produit par le denominateur.
			for k in range(len(p)):
				eq3bis[k] = eq1bis[k] * eq4bis[k] / denominateur

			r = eq3bis


			# Pour obtenir la proba d'avoir renforcement selon une cause k,
			# On fait un comptage du nombre de fois qu'on a eu renforcement lors de cette cause k
			for k in range(len(p)):
				pf = [0, 0]
				pf[0] = sum(((observations[_t][0] == 0) and (causesParticles[m][_t] == k)) for _t in range(t))
				pf[1] = sum(((observations[_t][0] == 1) and (causesParticles[m][_t] == k)) for _t in range(t))

				# Si la cause k a été choisie au moins une fois par la particule, on normalise les probas
				if(sum(pf) != 0):
					pf = [x / sum(pf) for x in pf]
				# Sinon, on estime que les probabilités d'avoir renforcement ou non pour cette cause
				# sont égales
				else:
					pf[0] = 0.5
					pf[1] = 0.5


				Vt[0] += r[k]*pf[0]
				Vt[1] += r[k]*pf[1]

		
		Vt = [x / M for x in Vt]

		print(Vt)

		print("-- NEW TRIAL --")

tabCauses = genCausesGaussienne(5)

tabProbas1 = genProbasObservations(tabCauses,1)
observations,causes = genObservations(tabCauses,tabProbas1,30)

particleFilter(observations)