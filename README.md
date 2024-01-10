# LAB1
Primer proyecto individual
<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1 - Mariela Bracamonte** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

¡Bienvenidos a la resolucion del primer proyecto individual de SoyHenry por Mariela Bracamonte, situándonos en el rol de un ***MLOps Engineer***.  

<hr>  

## **Introduccion (Contexto y rol a desarrollar)**

## Contexto

Tienes tu modelo de recomendación dando unas buenas métricas :smirk:, y ahora, cómo lo llevas al mundo real? :eyes:

El ciclo de vida de un proyecto de Machine Learning debe contemplar desde el tratamiento y recolección de los datos (Data Engineer stuff) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.


## Rol a desarrollar

Empezaste a trabajar como **`Data Scientist`** en Steam, una plataforma multinacional de videojuegos. El mundo es bello y vas a crear tu primer modelo de ML que soluciona un problema de negocio: Steam pide que te encargues de crear un sistema de recomendación de videojuegos para usuarios. :worried:

Vas a sus datos y te das cuenta que la madurez de los mismos es poca (ok, es nula :sob: ): Datos anidados, de tipo raw, no hay procesos automatizados para la actualización de nuevos productos, entre otras cosas… haciendo tu trabajo imposible :weary: . 

Debes empezar desde 0, haciendo un trabajo rápido de **`Data Engineer`** y tener un **`MVP`** (_Minimum Viable Product_) para el cierre del proyecto! Tu cabeza va a explotar 🤯, pero al menos sabes cual es, conceptualmente, el camino que debes de seguir :exclamation:. Así que espantas los miedos y pones manos a la obra :muscle:

<p align="center">
<img src="https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/DiagramaConceptualDelFlujoDeProcesos.png"  height=500>
</p>


## **Desarrollo del trabajo**

**`Transformaciones`**:  Se realizo un ETL de los tres subconjuntos que se encuentran [aca](https://github.com/mariebraca21/LAB1). Una vez extraidos los datos y desanidados los archivos, se procedio a la transformacion. En dicha transformacion, se validaron y corrigieron los formatos, eliminaron duplicados, completaron datos faltantes y seleccionaron las columnas pertinentes para poder continuar con el desarrollo del trabajo.

**`Feature Engineering`**:  En el dataset *user_reviews* se incluyen reseñas de juegos hechos por distintos usuarios. Mediante la libreria TextBlob se realizo un analisis de sentimientos de comentarios para evaluarlos segun una escala de 3 valores (0: Negativos, 1: Neutrales o faltantes y 2: Positivos) y almacenar la informacion en la columna ***'sentiment_analysis'*** .<br> El proceder de este trabajo se encuentra en [Fearure Engineering]

**`Preprocesamiento`**:  Previo a continuar con la disponibilizacion de la informacion analizada hasta ahora, se procede a realizar nuevas transformaciones a la informacion para poder simplificar los formatos y disminuir el peso de las bases de datos necesarias 

**`Desarrollo API`**:  Se utilizo la plataforma Render para poder disponibilizar las siguientes funciones consumibles desde una pagina web, y asi facilitar el acceso a informacion valiosa para toda la 'empresa':

+ def **PlayTimeGenre( *`genero` : str* )**: Devuelve el `año` con mas horas jugadas para dicho género.
  
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

+ def **UserForGenre( *`genero` : str* )**: Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf,
			     "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

+ def **UsersRecommend( *`año` : int* )**: Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **UsersWorstDeveloper( *`año` : int* )**: Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
  
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

+ def **sentiment_analysis( *`empresa desarrolladora` : str* )**: Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor. 

Ejemplo de retorno: {'Valve' : [Negative = 182, Neutral = 120, Positive = 278]}




**`Análisis exploratorio de los datos`**: _(Exploratory Data Analysis-EDA)_

Ahora cambiamos de rol y nos transformamos en un machine learning analyst, por ello, debemos realizar un EDA de la informacion recibida en este nuevo rol.
Para ello, extraemos nuevamente la informacion y analizamos las variables, si existen valores nulos, duplicados, outliers y como se distribuyen en general los valores de las principales variables para nuestro analisis.

Finalmente, se realiza una preparacion de los datos para el modelo de recomendacion en el que se genera un puntaje en funcion a un mix entre sentiment_analysis y recommend, generando una escala del 1 al 6 en relacion al puntaje del juego para el usuario, siendo 6 el mayor valor y 1 el menor.

Toda esta informacion se condensa con las bases de datos anteriores y simplifica nuevamente para poder alimentar el sistema de recomendacion.



**`Modelo de aprendizaje automático`**: 

Realizamos dos sistemas de recomendacion, segun las siguientes definiciones:

Si es un sistema de recomendación item-item:
+ def **recomendacion_juego( *`id de producto`* )**:
    Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

El sistema esta basado en una mezcla de recomendaciones por puntajes similares, es decir, si el juego tiene puntaje similar al solicitado; y tambien si los titulos son similares. Esto genera que los juegos tengan tendencia a recomendar precuelas-secuelas, hecho que resulta muy practico en el usuario final para poder mantener una direccion en cuanto a tipo de juego.

Si es un sistema de recomendación user-item:
+ def **recomendacion_usuario( *`id de usuario`* )**:
    Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

Este sistema esta basado en que juegos fueron bien puntuados por los usuarios que puntuaron juegos similarmente al usuario seleccionado, tendiendo a generar una comunidad de jugadores uniforme.

