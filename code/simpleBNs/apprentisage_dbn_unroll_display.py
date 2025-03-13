# Apprentissage d'un réseau bayésien dynamique avec pyAgrum
import pyAgrum as gum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Fonction pour générer un réseau bayésien avec k time slices
def generate_bn_k_slices(data, variables, k=2):
    """
    Génère un réseau bayésien avec k time slices.
    
    Args:
        data: DataFrame contenant les données
        variables: Liste des variables à inclure
        k: Nombre de time slices (par défaut: 2)
    
    Return:
        dbn_data qui est le DataFrame pour l'apprentissage du DBN
    """
    print(f"Génération d'un BN avec {k} tranches temporelles (#0 à #{k-1})...")
    
    # Sélection des variables
    data_selected = data[variables]
    
    # Création des tranches temporelles de 0 à k-1
    slices_data = []
    
    for i in range(k):
        # Décalage des données pour chaque tranche temporelle
        # Pour t0 (présent), pas de décalage
        # Pour t1, décalage de 1
        # Pour t2, décalage de 2
        # etc 
        if i == 0:
            slice_data = data_selected.copy().add_suffix(f'#{i}')
        else:
            slice_data = data_selected.shift(i).add_suffix(f'#{i}')
        
        slices_data.append(slice_data)
    
    # Combinaison de toutes les tranches dans un seul dataframe
    dbn_data = pd.concat(slices_data, axis=1)
    
    # Suppression des lignes avec valeurs manquantes
    dbn_data = dbn_data.dropna()
    
    return dbn_data

# Fonction pour discrétiser les données avec pyAgrum
def discretize_data_with_agrum(data, n_bins=2, method='quantile'):
    """
    Discrétise les données en utilisant pyAgrum.
    
    Args:
        data: DataFrame à discrétiser
        n_bins: Nombre de catégories (par défaut: 2)
        method: Méthode de discrétisation ('quantile', 'uniform', 'kmeans', 'NML', 'CAIM', 'MDLP')
    
    Returns:
        DataFrame discrétisé et template BN avec variables discrétisées
    """
    print(f"Discrétisation des variables en {n_bins} catégories avec pyAgrum (méthode: {method})...")
    
    # Import du module Discretizer de pyAgrum
    from pyAgrum.lib.discretizer import Discretizer
    
    # Création du discretizer avec les paramètres spécifiés
    discretizer = Discretizer(defaultDiscretizationMethod=method, defaultNumberOfBins=n_bins)
    
    # Création d'un BN template avec les variables discrétisées
    bn_template = discretizer.discretizedTemplate(data)
    
    # Création et remplissage du DataFrame discrétisé
    discretized_data = pd.DataFrame(index=data.index, columns=data.columns)
    
    for i, col in enumerate(data.columns):
        # Récupération de la variable discrète du BN template
        var = bn_template.variable(i)
        
        # Fonction de discrétisation qui convertit une valeur continue en indice d'intervalle
        def discretize_value(value):
            return var.index(str(value)) # var est bn_template.variable(i)
        
        # Application de la fonction à toute la colonne
        discretized_data[col] = data[col].map(discretize_value)
    
    return discretized_data, bn_template

def learn_dbn_with_slice_order(data, k=2, bn_template=None, n_bins=3):
    """Apprend un DBN avec sliceOrder pour respecter la causalité temporelle"""
        
    # Création d'un nouveau BN pour le DBN
    dbn = gum.BayesNet()
    
    # Identifier les variables pour chaque tranche temporelle
    slices = [[] for _ in range(k)]
    for col in data.columns:
        if '#' in col:
            parts = col.split('#')
            if len(parts) > 1 and parts[1].isdigit():
                t = int(parts[1])
                if t < k:
                    slices[t].append(col)
    
    # Ajouter toutes les variables au DBN avec les noms originaux (avec #)
    print(f"\nAjout des variables au DBN avec {n_bins} états...")
    for t in range(k):
        for var in slices[t]:
            # Ajouter la variable avec n_bins états (comme spécifié dans la discrétisation)
            dbn.add(var, n_bins)
            print(f"Variable ajoutée: {var}")
    
    # Apprendre la structure pour chaque tranche temporelle
    for t in range(k):
        print(f"\nApprentissage de la structure pour la tranche temporelle {t}...")
        slice_df = data[slices[t]]
        
        # Apprendre un BN pour cette tranche
        learner = gum.BNLearner(slice_df)
        learner.useScoreAIC()
        learner.useSmoothingPrior(1)
        bn = learner.learnBN()
        
        # Ajouter les arcs intra-tranche au DBN
        print(f"Ajout des arcs entre deux variables du meme time slice pour {t}...")
        for arc in bn.arcs():
            tail = bn.variable(arc[0]).name()
            head = bn.variable(arc[1]).name()
            
            try:
                dbn.addArc(dbn.idFromName(tail), dbn.idFromName(head))
                print(f"Arc ajouté: {tail} -> {head}")
            except Exception as e:
                print(f"Impossible d'ajouter l'arc: {tail} -> {head}, erreur: {str(e)}")
    
    # Ajouter des arcs inter-tranches (de t à t+1)
    if k > 1:
        print("\nAjout des arcs entre deux time slices differentes... ")
        for t in range(k-1):
            for var in slices[t]:
                base_name = var.split('#')[0]
                # Trouver la même variable dans la tranche suivante
                for next_var in slices[t+1]:
                    if next_var.split('#')[0] == base_name:
                        try:
                            dbn.addArc(dbn.idFromName(var), dbn.idFromName(next_var))
                            print(f"Arc inter-tranche ajouté: {var} -> {next_var}")
                        except Exception as e:
                            print(f"Impossible d'ajouter l'arc inter-tranche: {var} -> {next_var}, erreur: {str(e)}")
    
    return dbn

# Fonction pour "dérouler" (unroll) le DBN
def unroll_dbn(dbn, k):
    """
    Déroule un DBN en un réseau bayésien standard sur k time slices.
    
    Args:
        dbn: Réseau bayésien dynamique
        k: Nombre de tranches temporelles
    
    Returns:
        Le DBN unroll
    """
    print(f"Déroulement du DBN sur {k} time slices...")
    
    # Création d'un nouveau BN pour le DBN unroll
    unrolled_bn = gum.BayesNet()
    
    # Dictionnaire pour stocker les nœuds déjà créés dans le DBN unroll
    created_nodes = {}
    
    # Ajout des nœuds pour chaque tranche temporelle
    for t in range(k):
        for node in dbn.names():
            # Extraction du nom de base et du time slice
            if '#' in node:
                parts = node.split('#')
                base_name = parts[0]
                # Création d'un nouveau nom pour le nœud dans le DBN unroll
                new_node_name = f"{base_name}#{t}"
                
                # Vérifier si le nœud existe déjà
                if new_node_name not in created_nodes:
                    # Ajout du nœud au BN unroll avec le même nombre d'états
                    node_id = unrolled_bn.add(new_node_name, dbn.variable(node).domainSize()) 
                    created_nodes[new_node_name] = node_id
                else:
                    node_id = created_nodes[new_node_name]
                
    
    # Ajout des arcs en respectant les dépendances du DBN
    for arc in dbn.arcs():
        tail = dbn.variable(arc[0]).name()
        head = dbn.variable(arc[1]).name()
        
        tail_parts = tail.split('#')
        head_parts = head.split('#')
        
        if len(tail_parts) > 1 and len(head_parts) > 1:
            tail_base = tail_parts[0]
            head_base = head_parts[0]
            
            try:
                tail_slice = int(tail_parts[1])
                head_slice = int(head_parts[1])
                
                # Ajout des arcs correspondants dans le BN unroll
                for t in range(k-1):  # k-1 car la dernière tranche n'a pas de successeur
                    # Arcs dans la même tranche
                    if tail_slice == head_slice:
                        try:
                            unrolled_bn.addArc(
                                unrolled_bn.idFromName(f"{tail_base}#{t}"), 
                                unrolled_bn.idFromName(f"{head_base}#{t}")
                            )
                        except:
                            pass
                    
                    # Arcs entre tranches consécutives
                    if head_slice > tail_slice:
                        try:
                            unrolled_bn.addArc(
                                unrolled_bn.idFromName(f"{tail_base}#{t}"), 
                                unrolled_bn.idFromName(f"{head_base}#{t+1}")
                            )
                        except:
                            pass
            except:
                print(f"Impossible de déterminer les tranches temporelles pour {tail} -> {head}")
    
    return unrolled_bn

# Fonction pour afficher le BN
def display_bn(bn, filename="bn_graph"):
    """
    Affiche et sauvegarde une représentation graphique du réseau bayésien.
    
    Args:
        bn: Réseau bayésien à afficher
        filename: Nom du fichier pour sauvegarder l'image
    """    
    # Pour le .dot
    dot_str = bn.toDot()
    with open(f"{filename}.dot", "w") as f:
        f.write(dot_str)
    print(f"Représentation DOT sauvegardée dans {filename}.dot")
    
    # Utilisation de pyAgrum.lib.image pour générer une image PNG
    import pyAgrum.lib.image as gim
    gim.export(bn, f"{filename}.png")
    print(f"Image PNG générée: {filename}.png")
    
    # Affichage des informations sur le réseau
    print("\nInformations sur le réseau:")
    print(f"Nombre de nœuds: {bn.size()}")
    print(f"Nombre d'arcs: {bn.sizeArcs()}")
    print(f"Nœuds du réseau: {bn.names()}")

# Programme principal
if __name__ == "__main__":
    # Chargement des données
    print("Chargement des données...")
    data = pd.read_csv("DailyDelhiClimateTrain.csv")
    
    # Traitement de DataSet
    print("Traitement de DataSet...")
    # Conversion des dates et tri chronologique
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    
    # Sélection des variables
    variables = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
    
    # Nombre de tranches temporelles
    k = 3  # Pour le monent 3 time_slices
    
    # Génération du BN avec k tranches temporelles
    dbn_data = generate_bn_k_slices(data, variables, k)
    
    # Affichage des colonnes avant discretisation
    print("\nColonnes dans dbn_data:")
    print(dbn_data.columns.tolist())
    
    n_bins=3

    # Discrétisation des données avec le Discretizer de pyAgrum
    discretized_data, bn_template = discretize_data_with_agrum(dbn_data, n_bins, method='quantile')
    
    # Affichage des colonnes après discrétisation
    print("\nColonnes dans discretized_data:")
    print(discretized_data.columns.tolist())
    
    # Apprentissage du DBN
    print("\nApprentissage du DBN...")
    dbn = learn_dbn_with_slice_order(discretized_data, k, bn_template, n_bins)
    
    # Affichage du DBN
    display_bn(dbn, "delhi_climate_dbn")
    
    # Unroll le DBN
    unrolled_bn = unroll_dbn(dbn, k)
    
    # Affichage du BN unroll
    display_bn(unrolled_bn, "delhi_climate_unrolled_dbn")
