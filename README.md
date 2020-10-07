Cette section est reservee pour le traitement des textes. Tout d'abord, nous concatenons tous les fichiers java d'un apk dans un fichier appele output_file.txt avec la commande $ python3 concat.py. IL est a noter que cette commande s'execute sur un dossier (classes-dex2jar "vous devez donc dézipper classes-dex2jar.zip").\n\n
Apres cette phase, nous eliminons les comentaires dans le fichier resultat (output_file.txt) avec la commande $ python3 ./commentaire.py \n\n
Autre chose a faire est d'eliminer les ponctuations et les mots cles java dans le fichier resultat (apres elimination des commentaires) avec la commande $ python3 ./ponctuation.py \n\n

