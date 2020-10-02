import re
import string
mes_commentaires = ['@Override','/*','//']
a = string.punctuation
lines = open("sticker/a.txt",'r')
cmt = "false"
word_cmt = "/*"
fin_word_cmt = '*/'
debut_cmt = "//"
print(fin_word_cmt)
line = lines.readline()
with open("sticker/fichiers_traite.txt", "w+") as file:
    
    while line: #la premiere ligne
        
        if ((word_cmt in line) and (fin_word_cmt not in line)) :# si le caractere /* est dans la ligne et que */ n'y est pas
            line_commente = line.split()[:] # on split la ligne
            indice_d = line_commente.index('/*')
            indice_f = len(line_commente)
            for i in range(indice_d, indice_f) :
                if line_commente[i] in line:
                    line = line.replace(line_commente[i],'')
            file.write(line)
            cmt = "true"
            line = lines.readline()
            while (cmt == "true") :
                if (fin_word_cmt not in line):
                    line = line.replace(line, '')
                    line = lines.readline()
                else:
                    line_commente = line.split()[:] # on split la ligne
                    indice_fin_cmt = line_commente.index("*/")+1
                    for i in range(0, indice_fin_cmt):
                        if line_commente[i] in line:
                            line = line.replace(line_commente[i],'')
                    file.write(line)
                    cmt = "false"    
                    
        elif ((word_cmt in line) and (fin_word_cmt in line)): #lorsque le /* et */ sont dans la ligne
            line_commente = line.split()[:]
            indice_de = line_commente.index("/*")
            indice_fi = line_commente.index("*/")+1
            for i in range(indice_de, indice_fi):
                if line_commente[i] in line:
                    line = line.replace(line_commente[i],'')
            file.write(line)
        elif debut_cmt in line :
            line_commente = line.split()[:]
            indice_debut_cmt = line_commente.index('//')
            for i in range(indice_debut_cmt, len(line_commente)):
                if line_commente[i] in line:
                    line = line.replace(line_commente[i], '')
            file.write(line)
        else:
            file.write(line)
        line = lines.readline()
file.close
lines.close
