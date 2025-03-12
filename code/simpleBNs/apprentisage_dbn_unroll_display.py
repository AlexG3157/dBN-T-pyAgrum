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
    
    # Création d'un template BN avec les variables discrétisées
    bn_template = discretizer.discretizedTemplate(data)
    
    # Création et remplissage du DataFrame discrétisé
    discretized_data = pd.DataFrame(index=data.index, columns=data.columns)
    
    # Discrétisation simplifiée
    for i, col in enumerate(data.columns):
        # Récupération de la variable discrète du template
        var = bn_template.variable(i)
        
        # Fonction de discrétisation qui convertit une valeur continue en indice d'intervalle
        def discretize_value(value):
            return var.index(str(value)) # var est bn_template.variable(i)
        
        # Application de la fonction à toute la colonne
        discretized_data[col] = data[col].map(discretize_value)
    
    return discretized_data, bn_template

def learn_dbn_with_slice_order(data, k=2, bn_template=None, n_bins=3):
    """Apprend un DBN avec sliceOrder pour respecter la causalité temporelle"""
    print("\nUtilisation d'une approche manuelle pour l'apprentissage du DBN...")
    
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
    
    # Afficher les slices pour déboguer
    print("\nSlices organisées par tranche temporelle:")
    for i, slice_vars in enumerate(slices):
        print(f"Slice {i}: {slice_vars}")
    
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
        # Créer un sous-ensemble de données pour cette tranche
        slice_df = data[slices[t]]
        
        # Apprendre un BN pour cette tranche
        learner = gum.BNLearner(slice_df)
        learner.useScoreAIC()
        learner.useSmoothingPrior(1)
        bn = learner.learnBN()
        
        # Ajouter les arcs intra-tranche au DBN
        print(f"Ajout des arcs intra-tranche pour la tranche {t}...")
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
        print("\nAjout d'arcs inter-tranches pour modéliser les dépendances temporelles...")
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
    Déroule un DBN en un réseau bayésien standard sur k tranches temporelles.
    
    Args:
        dbn: Réseau bayésien dynamique
        k: Nombre de tranches temporelles
    
    Returns:
        Réseau bayésien déroulé
    """
    print(f"Déroulement du DBN sur {k} time slices...")
    
    # Création d'un nouveau BN pour le réseau déroulé
    unrolled_bn = gum.BayesNet()
    
    # Dictionnaire pour stocker les correspondances entre les nœuds du DBN et du BN déroulé
    node_mapping = {}
    
    # Dictionnaire pour stocker les nœuds déjà créés dans le BN déroulé
    created_nodes = {}
    
    # Ajout des nœuds pour chaque tranche temporelle
    for t in range(k):
        for node in dbn.names():
            # Extraction du nom de base et de la tranche temporelle
            if '#' in node:
                parts = node.split('#')
                base_name = parts[0]
                # Création d'un nouveau nom pour le nœud dans le BN déroulé
                new_node_name = f"{base_name}#{t}"
                
                # Vérifier si le nœud existe déjà
                if new_node_name not in created_nodes:
                    # Ajout du nœud au BN déroulé avec le même nombre d'états
                    node_id = unrolled_bn.add(new_node_name, dbn.variable(node).domainSize()) 
                    created_nodes[new_node_name] = node_id
                else:
                    node_id = created_nodes[new_node_name]
                
                # Stockage de la correspondance
                node_mapping[(node, t)] = node_id
    
    # Ajout des arcs en respectant les dépendances du DBN
    for arc in dbn.arcs():
        tail = dbn.variable(arc[0]).name()
        head = dbn.variable(arc[1]).name()
        
        # Extraction des informations de tranche
        tail_parts = tail.split('#')
        head_parts = head.split('#')
        
        if len(tail_parts) > 1 and len(head_parts) > 1:
            tail_base = tail_parts[0]
            head_base = head_parts[0]
            
            try:
                tail_slice = int(tail_parts[1])
                head_slice = int(head_parts[1])
                
                # Ajout des arcs correspondants dans le BN déroulé
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
                        if t+1 < k:
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
    print(f"Affichage du réseau bayésien ({bn.size()} nœuds, {bn.sizeArcs()} arcs)...")
    
    # Création d'un fichier DOT pour visualisation
    dot_str = bn.toDot()
    with open(f"{filename}.dot", "w") as f:
        f.write(dot_str)
    print(f"Représentation DOT sauvegardée dans {filename}.dot")
    
    # Utilisation de pyAgrum.lib.image pour générer une image PNG
    import pyAgrum.lib.image as gim
    gim.export(bn, f"{filename}.png")
    print(f"Image PNG générée avec pyAgrum.lib.image: {filename}.png")
    
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
    
    # Prétraitement pour DBN
    print("Prétraitement pour DBN...")
    # Conversion des dates et tri chronologique
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    
    # Sélection des variables climatiques
    variables = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
    
    # Nombre de tranches temporelles
    k = 3  # Par défaut, on utilise 3 tranches (t, t-1, t-2)
    
    # Génération du BN avec k tranches temporelles
    dbn_data = generate_bn_k_slices(data, variables, k)
    
    # Afficher les colonnes pour déboguer
    print("\nColonnes dans dbn_data:")
    print(dbn_data.columns.tolist())
    
    # Discrétisation des données avec pyAgrum
    discretized_data, bn_template = discretize_data_with_agrum(dbn_data, n_bins=3, method='quantile')
    
    # Afficher les colonnes après discrétisation
    print("\nColonnes dans discretized_data:")
    print(discretized_data.columns.tolist())
    
    # Apprentissage du DBN avec sliceOrder
    print("\nApprentissage du DBN avec sliceOrder...")
    # Utiliser la même valeur n_bins que celle utilisée pour la discrétisation
    n_bins = 3
    dbn = learn_dbn_with_slice_order(discretized_data, k, bn_template, n_bins)
    
    # Affichage du DBN
    display_bn(dbn, "delhi_climate_dbn")
    
    # Déroulement (unroll) du DBN
    unrolled_bn = unroll_dbn(dbn, k)
    
    # Affichage du BN déroulé
    display_bn(unrolled_bn, "delhi_climate_unrolled_dbn")
