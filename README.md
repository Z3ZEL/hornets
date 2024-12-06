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


# Video Extractor

```bash
poetry run extract path_to_bag_file path_to_output_folder (optional)frame_rate_to_skip
poetry run extract_vide path_to_bag_file path_to_output_folder 
poetry run extract_mp4 path_to_mp4 ../output/ (optional)frame_rate_to_skip (optional)range_time_code eg : 0:2,4:6
``` 

**frame_rate_to_skip** : skip some image to get less for instance if 2, it will skip one image out of two
**range_time_code** : extract only the part of the video between the time code given, the time code is in the format s:s or s:s,s:s,... to extract from 0 to 2 seconds and from 4 to 6 seconds you can use 0:2,4:6

**Note** : Be careful poetry run start from the project root, so if your video folder is above you need to specify ../path_to_bag_file for instance.

