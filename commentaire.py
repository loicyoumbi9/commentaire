import re 

exp = r'/\*.*\*/|//.*'

with open('output_file.txt', 'r') as f1, open('resultat.txt', 'w') as f2:

	# Reading the file and remove inline comments (i.e /*-----*/ or //----------)
	result = re.sub(exp, '', f1.read())

	# Preprocessing for remove multiline comments
	result = result.replace('\n', ')(').replace('*/', '*/\n')
	
	# Remove multiline comments (i.e /*--------
	#								   --------
	#								   --------*/)
	result = re.sub(exp, '', result)

	# Inversing the preprocessing
	result = result.replace('\n', '').replace(')(', '\n')
	
	# Storing the result in result.txt file
	f2.write(result)
with open('result.txt', 'r') as f3, open('resultat_traiter.txt', 'w+') as f4 :
	for line in f3:
		if not line.isspace():
            		f4.write(line)
