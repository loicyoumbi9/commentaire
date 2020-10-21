import re
from igraph import *
from numpy import *
from collections import defaultdict
import numpy as np
import pickle

# pour compter le nombre de neouds dans un graphe


def nomber_node():
    g = Graph.Read_GML("sticker/callgraph.gml")
    total_node = g.vcount()
    return total_node


# Algorithme de warshall

class MatriceWarshall(object):

    def __init__(self, path, node_number):
        self.node_number = node_number
        self.dictionnaire = {}
        donnees = open(path, "r").readlines()
        for line in donnees:
            ligne_splitee = line.split()
            self.dictionnaire[(int(ligne_splitee[0]),
                               int(ligne_splitee[1]))] = 1

        # Ce bout de code effectue une copie de fichier. Il faudra l'optimiser.
        # with open('sticker/paths_file.txt', 'w+') as file:
        #    for key in self.dictionnaire.keys():
        #        file.write(str(key[0])+','+str(key[1]) +"\n")

    def __getitem__(self, i, j):
        try:
            return self.dictionnaire[(i, j)]
        except KeyError:
            return 0

    def __setitem__(self, i, j, value):
        self.dictionnaire[(i, j)] = value

    def creer_chemin(self, i, j):
        self.dictionnaire[(i, j)] = 1

    def supprimer_connection(self, i, j):
        return

    def possible_paths(self):
        a = self.dictionnaire
        print(len(a))
        for k in range(self.node_number):
            for i in range(self.node_number):
                for j in range(self.node_number):
                    if True == self.__getitem__(i, j) or (self.__getitem__(i, k) and self.__getitem__(k, j)):
                        self.creer_chemin(i, j)
        a = self.dictionnaire
        print(len(a))

    def afficher(self):
        print(self.dictionnaire)

    def last_warshall_matrix(self, come_in, come_out):
        #i = 0
        #come_in_come_out = set(come_in).union(set(come_out))
        new_dict = {}
        for key in self.dictionnaire.keys():
            if key[0] in come_in and key[1] in come_out:
                new_dict[key] = self.__getitem__(key[0], key[1])
                #i += 1
        # print(i)
        #print("len new_dict", len(new_dict))
        self.dictionnaire = new_dict.copy()
        #print("len dictionnaire", len(self.dictionnaire))


def store_data(chemin):
    dbfile = open(chemin, 'ab')
    # source, destination
    pickle.dump(matrice.dictionnaire, dbfile)
    dbfile.close()


def load_data(chemin):
    # for reading also binary mode is important
    dbfile = open(chemin, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db


def noeud_extreme():
    line_tabulee = open("sticker/node_infos_final.txt", "r").readlines()
    liste_s = []
    liste_e = []

    for line_tab in line_tabulee:
        line_splitee = line_tab.split()
        if line_splitee[1] == '1':  # liste des external
            id_node = line_splitee[0]
            liste_s.append(id_node)

        if line_splitee[2] == '1':  # liste des internal
            id_node = line_splitee[0]
            liste_e.append(id_node)
    return (liste_e, liste_s)


def construire_noeud_extremes():
    i = 0
    entrypoint = []
    external = []
    nex_tab = []

    g = Graph.Read_GML("sticker/callgraph.gml")
    total_node = g.vcount()
    #g = defaultdict(list)
    # print(g)
    for i in range(0, total_node-1):
        a = g.get_all_shortest_paths(i)
        b = g.neighbors(i, mode="out")
        c = g.neighbors(i, mode="in")
        if not c:
            entrypoint.append(i)
        if not b:
            external.append(i)
    return(entrypoint, external)


if __name__ == "__main__":

    # pour extraire les sources et les targets
    lines = open("sticker/callgraph.gml", 'r').readlines()
    with open("sticker/source_target_edge.txt", "w+") as file:
    	for line in lines:
		    source = re.findall(r"source \w+", line)
		    target = re.findall(r"target \w+", line)
		    if len(source) > 0:
		        file.write(source[0]+" ")

		    if len(target) > 0:
		        file.write(target[0])
		        file.write("\n")

    # pour mettre les noeuds extraits dans un fichier
    lines = open("sticker/source_target_edge.txt", 'r').readlines()

    nodes = {}
    with open("sticker/source_target_edge_final.txt", "w+") as file:
        for line in lines:
            source = re.findall(r"\w+ target \w+", line)
            res = source[0].replace('target ', '')
            file.write(res + "\n")

    # pour avoir les informations sur les noeuds
    lines = open("sticker/callgraph.gml", 'r').readlines()
    with open("sticker/nodes_infos.txt", "w+") as file:
        for line in lines:
            label = re.findall(r'id \w+', line)
            external = re.findall(r"external \w+", line)
            entrypoint = re.findall(r"entrypoint \w+", line)
            if len(label) > 0:
                file.write(label[0]+" ")

            if len(external) > 0:
                file.write(external[0]+" ")

            if len(entrypoint) > 0:
                file.write(entrypoint[0])
                file.write("\n")

    mon_dictionnaire = {}
    lines = open("sticker/nodes_infos.txt", 'r').readlines()
    with open("sticker/node_infos_final.txt", "w+") as file:
        for line in lines:
            node_id = re.findall(r"\w+ external \w+", line)
            node_external = node_id[0].replace('external ', '')
            source = re.findall(r"entrypoint \w+", line)
            res = source[0].replace('entrypoint ', '')
            file.write(node_external+" "+res + "\n")

    path = "sticker/source_target_edge_final.txt"
    a = nomber_node()
    matrice = MatriceWarshall(path, a-1)

    matrice.afficher()
    matrice.possible_paths()
    matrice.afficher()

    matrice.dictionnaire = load_data('sticker/examplePickle.pickle')
    len(matrice.dictionnaire)
    e, s = construire_noeud_extremes()
    e = list(map(lambda x: int(x), e))
    s = list(map(lambda x: int(x), s))
    matrice.last_warshall_matrix(e, s)

    with open("sticker/new_graph.txt", "w+") as file:
        for element in matrice.dictionnaire:
            file.write(str(element[0])+' '+str(element[1])+"\n")
