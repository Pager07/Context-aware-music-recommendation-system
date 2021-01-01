## Music recommendation system using Facial Expression and Factorization Machines ## 
This project implements **context-aware recommendation system (CARS)**. 
CARS takes into consideration the contextual factors like mood,location and weather etc to recommend relevant songs to the user. Context is any additional information besides users, items and the ratings which may be relevant at the current time to make a recommendation. CARS aims to project users ‘U’ , items ‘I’ and context ‘C’ into feature space ‘F’. Then it generate recommendations based on the different similarities measures of these 3 entities in the feature space F.

It takes in account how users interact with the system within a particular “context”.This is important because the preferences for songs within one context may be different from those in another context.For Example, a user ‘X’ prefers a classical song ‘I’ when studying in a library. However, user X may also dislike the same item ‘I’ when outdoor.


![](./images/demo.gif)


**In the demo above, user's facial-expression is recorded overtime. The mean emotion is of 'SAD' class; The recommender system generates music recommendation with low-liveness accordingly.**

## Project objectives ## 
- Develop a context-aware music recommender system application
    - [x] Feature extraction and selection
        - [x] Consider user data 
        - [x] Consider item/song data 
        - [x] Selection a context (emotion/facial expression)
    - [x] 2D/MD recommender technique (user profiling/modelling) (Factorization Machines selected)
    - [x] Contextual recommendation generation
    - [x] Develop a simple UI for your application
        - [x] Enable system to recognise the active user (Webcam)
        - [x] Present predictions to user (Text upon clicking 'Generate prediction')

## HOW TO RUN THE PROJECT ##

-  ***PLEASE NOTE THAT THIS PROJECT REQUIRES WEBCAM***

- ***RUNNING THE SCRIPTS IN LAPTOP IS RECOMMENDED***

- To run the recommendation sys application:
- Download trained [Face dectection model and Factorization model here](https://drive.google.com/file/d/12xck0iK8K_dbDmjVSwBNWPydlQ635RBx/view?usp=sharing)
    - Put the folders **FaceDectectionModel** and **FMModel** inside the root **RECOMMENDER** folder
- Download preprocessed [#now-playing-RS dataset here](https://drive.google.com/file/d/1K_BR2Ucg3gTeXEM-u4cg7z-dGeYTKXtV/view?usp=sharing)
    - Put the folders **nowplayingRS** inside the root **RECOMMENDER** folder
- Install the libraries 
    - Basic libraries required are in the requirements.txt file
    - Factorization Machine library
        - use: pip install git+https://github.com/coreylynch/pyFM
        - info: https://github.com/coreylynch/pyFM
- Run the app.py



## Dataset ##
[nowplaying-RS: A Benchmark Dataset for Context Aware Music Recommendation](https://github.com/asmitapoddar/nowplaying-RS-Music-Reco-FM) is a large-scale benchmark dataset called #nowplaying-RS, which contains 11.6 million music listening events (LEs) of 139K users and 346K tracks collected from Twitter. The dataset comes with a rich set of item content features and user context features, and the timestamps of the LEs. Moreover, some of the user context features imply the cultural origin of the users, and some others—like hashtags.

- @inproceedings{smc18,
title = {#nowplaying-RS: A New Benchmark Dataset for Building Context-Aware Music Recommender Systems},
author = {Asmita Poddar and Eva Zangerle and Yi-Hsuan Yang},
url = {http://mac.citi.sinica.edu.tw/~yang/pub/poddar18smc.pdf},
year = {2018},
date = {2018-07-04},
booktitle = {Proceedings of the 15th Sound & Music Computing Conference},
address = {Limassol, Cyprus},
note = {code at https://github.com/asmitapoddar/nowplaying-RS-Music-Reco-FM},
tppubtype = {inproceedings}
}