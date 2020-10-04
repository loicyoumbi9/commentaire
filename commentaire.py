import re 

exp = r'/\*.*\*/|//.*'

with open('a.txt', 'r') as f1, open('result.txt', 'w') as f2:

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
