import re
import string
a = string.punctuation
ma_ponctuation = ['if', 'else','final', 'else if','private','android ','import ','abstract ','package ','class ','return', 'public ','static','void ','boolean ','String ','null']
lines = open("resultat.txt",'r').readlines()
with open("resultat_ponctue.txt", "w+") as file:
	for line in lines:
		if line != '\n':
			for word in ma_ponctuation:
				if word in line:
					line = line.replace(word, '')
			if line != '\n':
				new_line = " ".join("".join([" " if ch in string.punctuation else ch for ch in line]).split())
				file.write(new_line +"\n")
with open('resultat_ponctue.txt', 'r') as f3, open('resultat_ponctue_final.txt', 'w+') as f4 :
	for line in f3:
		if not line.isspace():
            		f4.write(line)
