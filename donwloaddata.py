import os
import re
import urllib.request
import math
import shutil
import zipfile
from zipfile import ZipFile
from click._compat import raw_input


#url = urllib.request.Request('file:/Users/cpicon/PycharmProjects/KiwiC/FullIJCNN2013.zip')
url = urllib.request.Request('http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip')
try:
    req = urllib.request.urlopen(url)
except urllib.error.HTTPError as e:
    print(e.code)
    print(e.read())


    # nombre archivo url
name = 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'

    # Nombre del archivo a partir del URL
filename = name[name.rfind("/") + 1:]
while not filename:
        filename = raw_input("No se ha podido obtener el nombre del "
                             "archivo.\nEspecifique uno: ")
print( "Descargando %s..." % filename)
    #Archivo local
f = open(filename, "wb")
    #Escribir en un nuevo fichero local los datos obtenidos vía HTTP.
f.write(req.read())
    #Cerrar ambos
f.close()
req.close()
print("%s descargado correctamente." % filename)

def copy( zip_ref, files, target_folder ):
    """ Copy a  the files to an specific folder """

    print("Copying %s files to %s" % (len(files), target_folder))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file in files:
        filename = os.path.basename(file)
        source = zip_ref.open(file)
        target = open(target_folder + filename, 'wb')
        with source, target: shutil.copyfileobj(source, target)


PATH = os.path.abspath('.')
zpr = zipfile.ZipFile('/Users/cpicon/PycharmProjects/KiwiC/FullIJCNN2013.zip', 'r')
carpeta = []
for i in zpr.namelist():
    if re.match(r"\A{}/\d+/\Z".format("FullIJCNN2013"), i):
        carpeta.append(i)
    for ID in carpeta:
        class_files = [file for file in zpr.namelist()
                       if re.match(r"\A{}\d+.ppm\Z".format(ID), file)]

        print("\nClass folder {} contains {} files".format(ID, len(class_files)))
        test_files = class_files[:math.ceil(0.2 * len(class_files))]
        train_files = class_files[math.ceil(0.2 * len(class_files)):]

        train_folder = ID.replace("FullIJCNN2013", PATH + "/images/train")
        copy(zpr, train_files, train_folder)

        test_folder = ID.replace("FullIJCNN2013", PATH + "/images/test")
        copy(zpr, test_files, test_folder)


