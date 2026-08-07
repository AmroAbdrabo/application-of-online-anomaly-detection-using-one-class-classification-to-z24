

# Código para el artículo sobre "Aplicación de Detección de Anomalías mediante Clasificación de una Clase al Puente Z24"
## Descripción general

Este repositorio contiene el código y los recursos para el proyecto de tesis de maestría de Amro. El proyecto tiene como objetivo investigar e implementar diversas técnicas para evaluar el estado de daño de un puente.

## Introducción
Modos del puente Z24. Todos los apoyos se eligen como apoyos fijos. Se seleccionan apoyos fijos en las bases de los pilares, así como en los dos extremos del puente que se fusionan con la carretera. 
![Formas modales](https://drive.usercontent.google.com/download?id=12GrWRltz42P_b4djttd1gvW2yIQ7KqEy&export=view&authuser=0)

## Requisitos (Capítulos 2 y 3)
- <b> Python 3.11.5 </b>

<div align="center">

| Paquete     | Versión              |
|-------------|----------------------|
| numpy       | 1.26.1               |
| matplotlib  | 3.7.2                |
| sklearn     | 1.3.0                |
| xgboost     | 1.7.6                |
| pandas      | 2.0.3                |
| scipy       | 1.11.2               |
| seaborn     | 0.12.2               |
| tsfresh     | 0.20.1               |
| skrebate    | 0.62                 |
| river       | 0.19.0               |
| plotly      | 5.17.0               |

</div>

- <b> Variable de sistema Path (Windows) </b> 
<div align="center">

![Variable de sistema](https://drive.usercontent.google.com/download?id=1GjgFIP7-BKzdv5xZ_BG8s1A3C_Arkjcf&export=view&authuser=0) 

</div>

Para usuarios de MacOS, consulte https://phoenixnap.com/kb/set-environment-variable-mac.

Los únicos dos archivos que deben ejecutarse son eda.ipynb en la carpeta Chapter2-Z24-dataset y river_experiments_occ.ipynb en la carpeta Chapter3-ActiveLearning. Descargue los archivos de datos Z24 para Chapter2-Z24-dataset desde https://polybox.ethz.ch/index.php/s/8T6Lu8Hi8VqJcze y extráyalos para que exista la carpeta /data/, cuyo contenido son las carpetas del 01 al 17. Establezca la variable de entorno del sistema para que apunte a la ruta de la carpeta /data/. Reemplace path con la ruta en su computadora de la carpeta data. 

Una vez que se haya ejecutado eda.ipynb, debería obtener como salida X_train_new.npy, labels_train_new.npy, X_test_new.npy y labels_test_new.npy. Guárdelos dentro de una carpeta llamada /features dentro del directorio referenciado por la variable de entorno Z24_DATA. También puede encontrar estos archivos numpy aquí https://polybox.ethz.ch/index.php/s/IdGWA8OKFVE0lfa. A continuación, ejecute el cuaderno river_experiments_occ_new.ipynb (dentro de Chapter3-ActiveLearning) para obtener los resultados para el segmento de aprendizaje en línea del proyecto. 

## Generación de Datos de Vibración Simulados con Ansys (Capítulo 4)

Para obtener los datos generados en Excel de la respuesta en frecuencia en Chapter4-PhysicalSimulation, siga este video tutorial https://polybox.ethz.ch/index.php/s/iAWIzudH6P3gF8K.
Tenga en cuenta que, en el tutorial, solo lo hago para unos pocos canales, aunque hacerlo para todos los canales es autoexplicativo. 
![Captura de pantalla de Ansys](https://drive.usercontent.google.com/download?id=1Ig5SJIwKs5HkKpB3Jd53_PWp2A9bHNTi&export=view&authuser=0)

## Instalación
Los cuadernos se ejecutaron dentro de Visual Studio Code, donde el directorio de trabajo contiene las carpetas de todos los capítulos. Para Ansys 2023 WB, la instalación junto con las instrucciones se puede encontrar en la ETH IT Shop https://itshop.ethz.ch/ (se requiere una conexión VPN para ejecutarlo)
## Uso
## Contribuciones
## Agradecimientos
Dr. Cyprien Hoelzl, Dr. Yves Reuland, Prof. Dr. Eleni Chatzi
