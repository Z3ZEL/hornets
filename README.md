# Hornets
# Cahier des charges

## Description du besoin client
Le problème plus général du client est de faire en sorte que les abeilles n'aient plus peur de sortir de la ruche pour aller butiner. En effet, les abeilles ont peur des frelons asiatiques qui les attaquent lorsqu'elles sortent de la ruche. Le client souhaite donc trouver une solution pour protéger les abeilles des frelons asiatiques afin qu'elles puissent emagaziner des ressources pour passer l'hiver.

Plus spécifiquement, dans le cadre de notre projet, et de l'installation actuel, la problématique est de détecter les frelons asiatiques sur la caméra installée à l'entrée de la ruche, afin de les détruire avec un laser. Cette détection doit se faire avec un minimum de latence afin de ne pas manquer le tir.
## Verrous technologiques
Dans notre cas précis, un des principaux verrous technologiques est de différentier les frelons des abeilles, en effet, en utilisant un modèle de deep learning nous ne pouvons pas savoir à l'avance la capacité de notre modèle à différencier les frelons des abeilles avec les données que nous avons à disposition.

## Description du prototype 
On propose de faire un pipeline en deux parties : une première partie dans laquelle on réalise un traitement de l’image, puis une deuxième partie dans laquelle une IA analysera l’image traitée.

## Description de la solution proposée long terme
La solution prototypée pendant les 12 heures de cours pourra être amélioré en agrandissant le dataset de l'IA que nous allons entrainer, car il sera forcément limité par les données fournies dans le cadre de ce projet, ou en améliorant l'algorithme de traitement d'image.


