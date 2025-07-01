# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:30:52 2022

@author: ypatia
"""



#replace strings with other strings in file 
 
fin = open("protein_all.txt", "rt")
fout = open("protein.txt", "wt")

for line in fin:
    #for s in line:
    fout.write(line.replace(": .", ":."))
	
fin.close()
fout.close()

#remove strings in file

"""fin = open("data.txt", "rt")
fout = open("out.txt", "wt")

for line in fin:
	fout.write(' '.join(line.split()))
	
fin.close()
fout.close()"""

"""import re

fin = open("protein.txt", "rt")
fout = open("protein1.txt", "wt")

for line in fin:
	fout.write(re.sub('\s+',' ',line))
	
fin.close()
fout.close()"""