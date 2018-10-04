import click
import os
import donwloaddata
import cv2 as cv
import imghdr
import numpy as np
from PIL import Image, ImageDraw
from models import model1
from models import model3
from utils import LABELS

modelos = {"model1": model1, "model2": 0, "model3": model3}
LABELS = LABELS()
# download command
##################
@click.group()
def data():
    pass


@data.command()
@click.option('-p', required=True, type=click.Choice(['model1', 'model2', 'model3']), help= 'descarga los datos necesarios para los modelos')
def download(p):
    if p==p:
        donwloaddata()
    else:
        click.echo('parametro no valido')


# train command
###############
@click.group()
def training():
    pass


@training.command()
@click.option('-m', required=True, type=click.Choice(['model1', 'model2', 'model3']))
@click.option('-d', default=os.path.join('images', 'train'), type=click.STRING, help='Directory with training data')
def train( m, d ):
    """Usar con la siguiente sintaxis: train -m modelo -d path/to/folder """
    lenet = modelos[m] == model3
    clase = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]
    xdata = []
    ydata = []
    for ID in clase:
        class_folder = os.path.join(d, ID)
        for file in os.listdir(class_folder):
            if os.path.basename(os.path.join(class_folder, file)):  # images only
                imagePath = os.path.join(os.path.abspath('.'), d, ID, file)

                image = cv.imread(imagePath, 0)  #  se carga la imagen en escala de grises
                size = (32, 32) if lenet else (28, 28)
                resized = cv.resize(image, size)  # cv2 drops channel dimension
                xdata.append(resized[..., None] if lenet else resized.flatten())
                ydata.append(int(ID))
                print(len(ydata))
    if m == 'model1':
        model = model1.LogRskLearn()
        model.train(xdata, ydata)
    elif m == 'model2':
        pass
    else:
        model = model3.LeNet5()
        model.train(xdata, ydata)


# test command
##############
@click.group()
def testing():
    pass


@testing.command()
@click.option('-m', required=True, type=click.Choice(['model1', 'model2', 'model3']))
@click.option('-d', required=True)
def test( m, d ):
    """Usar con la siguiente sintaxis: test -m modelo -d path/to/folder """
    lenet = modelos[m] == model3

    clase = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]
    xdata = []
    ydata = []
    for ID in clase:
        class_folder = os.path.join(d, ID)
        for file in os.listdir(class_folder):
            if os.path.basename(os.path.join(class_folder, file)):  # images only
                imagePath = os.path.join(os.path.abspath('.'), d, ID, file)

                image = cv.imread(imagePath, 0)  # #  se carga la imagen en escala de grises
                size = (32, 32) if lenet else (28, 28)
                resized = cv.resize(image, size)  # cv2 drops channel dimension
                xdata.append(resized[..., None] if lenet else resized.flatten())
                ydata.append(int(ID))
    if m == 'model1':
        model = model1.LogRskLearn()
        model.accuracy(xdata, ydata)
    elif m == 'model2':
        pass
    else:
        model = model3.LeNet5()
        model.accuracy(xdata, ydata)



# infer command
##############
@click.group()
def infering():
    pass


@infering.command()
@click.option('-m', required=True, type=click.Choice(['model1', 'model2', 'model3']))
@click.option('-d', required=True,  help='Directory with training data')
def infer( m, d ):
    """Usar con la siguiente sintaxis: infer -m modelo -d path/to/folder """
    global predictions

    lenet = modelos[m] == model3
    xdata = []
    files = []
    data_dir = os.path.join(d)
    print("Nuevas imagenes de : ", data_dir)
    for file in os.listdir(d):
        imagePath = os.path.join(os.path.abspath('.'), data_dir, file)
        if os.path.basename(imagePath) and imghdr.what(imagePath):
            image = cv.imread(imagePath, 0)  # #  se carga la imagen en escala de grises
            size = (32, 32) if lenet else (28, 28)
            resized = cv.resize(image, size)
            xdata.append(resized[..., None] if lenet else resized.flatten())
            files.append(imagePath)

    xdata = np.array(xdata)
    if m == 'model1':
        model = model1.LogRskLearn()
        predictions = model.predict(xdata)
    elif m == 'model2':
        pass
    else:
        model = model3.LeNet5()
        predictions = model.predict(xdata)

    click.echo("Predicciones :")
    for i in range(len(predictions)):

        class_label = LABELS[str(predictions[i])]
        click.echo('%s'  %class_label)

        img = Image.open(files[i])# se abre la imagen del archivo i
        img = img.resize((200, 200))# se reajusta el tamano
        draw = ImageDraw.Draw(img)
        draw.line((0, 0) + (200, 0), fill=255, width=20)
        draw.text((0, 0), class_label, fill=(0, 0, 0))

        img.show()


start = click.CommandCollection(sources=[training, testing, infering])

if __name__ == '__main__':
    start()