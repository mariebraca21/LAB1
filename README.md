LABORATORIO1
Primer proyecto individual



PROYECTO INDIVIDUAL Nº1 - Mariela Bracamonte
Machine Learning Operations (MLOps)


¡Bienvenidos a la resolución del primer proyecto individual de SoyHenry por Mariela Bracamonte, situándonos en el rol de un MLOps Engineer .

Introducción (Contexto y rol a desarrollar)
Contexto
Tienes tu modelo de recomendación dando unas buenas métricas 😏, y ahora, ¿cómo lo llevas al mundo real? 👀

El ciclo de vida de un proyecto de Machine Learning debe contemplar desde el tratamiento y recolección de los datos (Data Engineer things) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.

Rol a desarrollar
Empezá a trabajar como Data Scientisten Steam, una plataforma multinacional de videojuegos. El mundo es bello y vas a crear tu primer modelo de ML que soluciona un problema de negocio: Steam pide que te encargues de crear un sistema de recomendación de videojuegos para usuarios. 😟

Vas a sus datos y te das cuenta que la madurez de los mismos es poca (ok, es nula 😭 ): Datos anidados, de tipo raw, no hay procesos automatizados para la actualización de nuevos productos, entre otras cosas… haciendo tu trabajo imposible 😩 .

¡Debes empezar desde 0, haciendo un trabajo rápido de Data Engineery tener un MVP( Producto Mínimo Viable ) para el cierre del proyecto! Tu cabeza va a explotar 🤯, pero al menos sabes cual es, conceptualmente, el camino que debes de seguir ❗. Así que espantas los miedos y pones manos a la obra 💪



Desarrollo del trabajo
Transformaciones: Se realiza un ETL de los tres subconjuntos que se encuentran aquí . Una vez extraidos los datos y desanados los archivos, se procedio a la transformacion. En dicha transformación, se validaron y corrigieron los formatos, eliminaron duplicados, completaron los datos faltantes y seleccionaron las columnas pertinentes para poder continuar con el desarrollo del trabajo.

Feature Engineering: En el conjunto de datos user_reviews se incluyen reseñas de juegos hechos por distintos usuarios. Mediante la librería TextBlob se realiza un análisis de sentimientos de comentarios para evaluarlos según una escala de 3 valores (0: Negativos, 1: Neutrales o faltantes y 2: Positivos) y almacenar la información en la columna ' sentiment_analysis' .
El procedimiento de este trabajo se encuentra en [Fearure Engineering]

Preprocesamiento: Previo a continuar con la disponibilizacion de la información analizada hasta ahora, se procede a realizar nuevas transformaciones a la información para poder simplificar los formatos y disminuir el peso de las bases de datos necesarias.

Desarrollo API: Se utiliza la plataforma Render para poder disponibilizar las siguientes funciones consumibles desde una página web, y asi facilitar el acceso a información valiosa para toda la 'empresa':

def PlayTimeGenre( genero: str ) : Devuelve el añocon más horas jugadas para dicho género.
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

def UserForGenre( genero: str ) : Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas : 23}]}

def UsersRecommend( año: int ) : Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def UsersWorstDeveloper( año: int ) : Devuelve el top 3 de desarrolladores con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

def sentiment_analysis( empresa desarrolladora: str ) : Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentran categorizados con un análisis de sentimiento como valor.
Ejemplo de retorno: {'Valve' : [Negativo = 182, Neutro = 120, Positivo = 278]}

Análisis exploratorio de los datos: (Análisis de datos exploratorios-EDA)

Ahora cambiamos de rol y nos transformamos en un analista de aprendizaje automático, por ello, debemos realizar un EDA de la información recibida en este nuevo rol. Para ello, extraemos nuevamente la información y analizamos las variables, si existen valores nulos, duplicados, outliers y como se distribuyen en general los valores de las principales variables para nuestro análisis.

Finalmente, se realiza una preparación de los datos para el modelo de recomendación en el que se genera un puntaje en función a una mezcla entre sentiment_analysis y recomend, generando una escala del 1 al 6 en relación al puntaje del juego para el usuario, siendo 6 el mayor valor y 1 el menor.

Toda esta información se condensa con las bases de datos anteriores y simplifica nuevamente para poder alimentar el sistema de recomendación.

Modelo de aprendizaje automático:

Realizamos dos sistemas de recomendación, según las siguientes definiciones:

Si es un sistema de recomendación artículo-artículo:

def recomendacion_juego( id de producto) : Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.
El sistema está basado en una mezcla de recomendaciones por puntajes similares, es decir, si el juego tiene puntaje similar al solicitado; y tambien si los titulos son similares. Esto genera que los juegos tengan tendencia a recomendar precuelas-secuelas, hecho que resulta muy práctico en el usuario final para poder mantener una dirección en cuanto a tipo de juego.

Si es un sistema de recomendación user-item:

def recomendacion_usuario( id de usuario) : Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.
Este sistema esta basado en que juegos fueron bien puntuados por los usuarios que puntuaron juegos similarmente al usuario seleccionado, tendiendo a generar una comunidad de jugadores uniforme.
