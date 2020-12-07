# DSS-TP-Final

### Para levantar la app:
1. Ejecutar el comando ```xhost +``` para habilitar la compartición de dispositivos (Necesario para el acceso a la camara desde el container).
2. Ejecutar el comando ```docker-compose up -d```
3. Funcionamientos:
    - Para ejecutar el entrenamiento de los modelos:
        1. Copiar los archivos de las bases de datos de google, faces y fer a la carpeta ```/app/datos/datasets```
        2. Ejecutar el comando ```docker exec -it dss_tp_final python command.py init```
        - Para obtener la ayuda del script helper de comandos ejecutar ```docker exec -it dss_tp_final python command.py h``` (Los modelos disponibles son 'rf', 'svm' y 'cnn')
        - El entrenamiento de alguno de los modelos se ejecuta con el argumento ```docker exec -it dss_tp_final python command.py <modelo> fit``` (Los modelos disponibles son 'rf', 'svm', 'cnn')
        - La prueba de predicción de alguno de los modelos se ejecuta con el argumento ```docker exec -it dss_tp_final python command.py <modelo> predict <path_imagen>``` (Los modelos disponibles son 'rf', 'svm', 'cnn')
    - Para ejecutar la aplicación web:
        1. Acceder a la url ```localhost:5000``` desde cualquier navegador
