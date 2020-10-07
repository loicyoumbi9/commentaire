#/bin/sh
#je dois etre dans le dossier dex2jar pour l'executer
for fichier in *.apk
do	
	a="${fichier%%.*}"
	androguard cg "$fichier"
	androguard axml -i "$fichier" -o "$a".xml
	xml = "$a".xml
	mkdir "$a"
	cp "$fichier" "$a"/
	cd "$a"
	unzip "$fichier" 
	sudo chmod -R 777 META-INF &
	sudo chmod -R 777 res &
	sudo chmod -R 777 AndroidManifest.xml &
	sudo chmod -R 777 *.dex &
	sudo chmod -R 777 resources.arsc
	#cp -v $(find -name *.dex) ../
	for dexjar in *.dex
	do
		./../d2j-dex2jar.sh "$dexjar"
	done
	
	 
	#cp -v $(find -name *jar) "$a"/
	for jar_file in *.jar
	do
		./../../jadx/build/jadx/bin/jadx "$jar_file"
	done
	rm -f *.jar
	rm -f *.apk
	for nom_dossier in classes* 
	do
		if [ -d "$nom_dossier" ]; then 
			cd "$nom_dossier"
			rm -R resources
		fi
	done
	cd ..
	mkdir ../../java_apk_decompiles/"$a"
	for nom_dossier_a_copier in classes* 
	do
		if [ -d "$nom_dossier_a_copier" ]; then 
			cp -r "$nom_dossier_a_copier" ../../java_apk_decompiles/"$a"/
		fi
	done
	
	cd ..
	cp -v callgraph.gml ../java_apk_decompiles/"$a"/
	cp -v "${fichier%%.*}".xml ../java_apk_decompiles/"$a"/
	rm -f callgraph.gml
	rm -f *.dex
	rm -f "$fichier"
	rm -f "${fichier%%.*}".xml
	cp -v fichies.sh ../java_apk_decompiles/"$a"/
	
	cd ../java_apk_decompiles/"$a"/
	./fichies.sh
	cd ../../dex2jar
done
