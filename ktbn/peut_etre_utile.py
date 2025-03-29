import pyAgrum as gum
import pandas as pd
import numpy as np
from pyAgrum.lib.discretizer import Discretizer
import os

class KTBNLearner:
    """
    Classe pour l'apprentissage d'un KTBN.
    """
    
    def __init__(self, k, n_bins, discretization_method='quantile'):
        """
        Constructeur de base pour KTBNLearner.
        
        Args:
            k (int): nombre de time slices
            n_bins (int): nombre de catégories pour la discrétisation
            discretization_method (str): méthode de discrétisation ('quantile', 'uniform', 'k-means', etc)
        """
        self.k = k
        self.n_bins = n_bins
        self.discretization_method = discretization_method
    
    def learn_from_dataframes(self, dataframes, variables=None):
        """
        Apprend un KTBN à partir d'une liste de DataFrames.
        
        Processus:
        1) Création d'une base de 0 à k-1
        2) Apprentissage avec des contraintes via sliceOrder
        3) Apprentissage de la transition avec des contraintes via sliceOrder
        
        Args:
            dataframes (list): liste de DataFrames pandas pour chaque time slice
            variables (list): liste des noms de variables à inclure (si None, utilise toutes les colonnes)
            
        Returns:
            KTBN: une nouvelle instance de KTBN
        """
        from KTBN import KTBN
        
        if variables is None:
            variables = dataframes[0].columns.tolist()
        
        # Préparation les données
        prepared_data = self._prepare_data_from_dataframes(dataframes, variables)
        
        # Discrétise
        discretized_data = self._discretize_data(prepared_data)
        
        # Apprend
        bn, temporal_variables = self._learn_structure(discretized_data, variables)
        
        # Créer le KTBN appris
        ktbn = KTBN(k=self.k, variables=variables, n_bins=self.n_bins, discretization_method=self.discretization_method)
        ktbn.bn = bn
        ktbn.temporal_variables = temporal_variables
        
        return ktbn
    
    def learn_from_csv_files(self, csv_files, variables=None):
        """
        Apprend un KTBN à partir d'une liste de fichiers CSV.
        
        Args:
            csv_files (list): liste comportant le nom des fichiers csv (1 csv par time-slice)
            variables (list): liste des noms des variables à inclure
            
        Returns:
            KTBN: une nouvelle instance de KTBN
        """
        # lire les csv avec pandas
        dataframes = [pd.read_csv(file) for file in csv_files]
        
        # J'utilise la méthode pour les DataFrames
        return self.learn_from_dataframes(dataframes, variables)
    
    def _prepare_data_from_dataframes(self, dataframes, variables):
        """
        Prépare les données pour l'apprentisage d'un KTBN à partir d'une liste de DataFrames(1 DataFrame par time-slice)
        
        Args:
            dataframes (list): liste de DataFrames pour chaque tranche temporelle
            variables (list): liste des noms de variables à inclure
            
        Returns:
            DataFrame: DataFrame préparé avec les variables temporelles
        """
        # Vérifier que nous avons suffisamment de DataFrames
        if len(dataframes) < self.k:
            raise ValueError(f"Nombre insuffisant de DataFrames. Attendu: {self.k}, Reçu: {len(dataframes)}")
        
        # Sélectionner les variables dans chaque DataFrame
        selected_dfs = []
        for i, df in enumerate(dataframes[:self.k]):
            # Sélectionner les variables spécifiées
            selected_df = df[variables].copy()
            
            # Ajouter le suffixe de tranche temporelle
            selected_df = selected_df.add_suffix(f'#{i}')
            
            selected_dfs.append(selected_df)
        
        # Combiner les DataFrames
        prepared_data = pd.concat(selected_dfs, axis=1)
        
        return prepared_data
    
    def _discretize_data(self, data):
        """
        Discrétise les données en utilisant pyAgrum.
        
        Cette méthode utilise le module Discretizer de pyAgrum pour transformer
        les variables continues en variables discrètes avec le nombre de catégories
        et la méthode de discrétisation spécifiés.
        
        Args:
            data (DataFrame): DataFrame avec des variables continues
            
        Returns:
            DataFrame: DataFrame avec des variables discrétisées
        """
        print(f"Discrétisation des variables en {self.n_bins} catégories avec pyAgrum (méthode: {self.discretization_method})...")
        
        # Création du discretizer avec les paramètres spécifiés
        discretizer = Discretizer(defaultDiscretizationMethod=self.discretization_method, 
                                 defaultNumberOfBins=self.n_bins)
        
        # Création d'un template BN avec les variables discrétisées
        bn_template = discretizer.discretizedTemplate(data)
        
        # Création et remplissage du DataFrame discrétisé
        discretized_data = pd.DataFrame(index=data.index, columns=data.columns)
        
        for i, col in enumerate(data.columns):
            # Récupération de la variable discrète du template
            var = bn_template.variable(i)
            
            # Fonction de discrétisation qui convertit une valeur continue en indice d'intervalle
            def discretize_value(value):
                try:
                    return var.index(str(value))
                except:
                    return 0  # Valeur par défaut en cas d'erreur
            
            # Application de la fonction à toute la colonne
            discretized_data[col] = data[col].map(discretize_value)
        
        return discretized_data
    
    def _learn_structure(self, data, variables):
        """
        Apprend la structure du KTBN à partir des données discrétisées en utilisant sliceOrder.
        
        Cette méthode suit les instructions:
        1) Création d'une base de 0 à k-1
        2) Apprentissage avec des contraintes via sliceOrder
        3) Apprentissage de la transition avec des contraintes via sliceOrder
        4) Retour d'un KTBN
        
        Args:
            data (DataFrame): DataFrame discrétisé
            variables (list): liste des noms de variables (sans le #)
            
        Returns:
            tuple: (gum.BayesNet, list) Le réseau bayésien appris et la liste des variables temporelles
        """
        # 1) Identifier les variables pour chaque time sclice
        slices = [[] for _ in range(self.k)]
        
        for col in data.columns:
            if '#' in col:
                parts = col.split('#')
                if len(parts) == 2 and parts[1].isdigit():
                    t = int(parts[1])
                    if t < self.k:
                        slices[t].append(col)
        
        # 2) Créer un template pour le BN avec les variables correctes
        template = gum.BayesNet()
        
        # Ajouter toutes les variables au template
        for var in variables:
            for t in range(self.k):
                var_name = f"{var}#{t}"
                # Utiliser RangeVariable au lieu de LabelizedVariable
                template.add(gum.RangeVariable(var_name, var_name, 0, self.n_bins-1))
        
        # 3) Créer un fichier CSV temporaire avec des options spécifiques
        temp_csv_path = "temp_data.csv"
        
        # Utiliser des options spécifiques pour to_csv
        data.to_csv(temp_csv_path, index=False, quoting=1)  # quoting=1 pour QUOTE_ALL
        
        # 4) Créer un BNLearner avec le fichier CSV et le template
        learner = gum.BNLearner(temp_csv_path, template)
        
        # 5) Configurer le BNLearner pour l'apprentissage de DBN
        learner.useMIIC()  # Utiliser explicitement l'algorithme MIIC
        learner.useNMLCorrection()
        learner.useSmoothingPrior(1)
        
        # 6) Définir l'ordre des time slices setSliceOrder
        learner.setSliceOrder(slices)
        
        # 7) Apprendre la structure
        bn = learner.learnBN()
        
        # Créer la liste des variables temporelles
        temporal_variables = []
        for t in range(self.k):
            for var in variables:
                temporal_variables.append(f"{var}#{t}")
        
        return bn, temporal_variables
