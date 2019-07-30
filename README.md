# Machine-Learning-CO2-Emission
A dataset containting a variety of car features was downloaded from  [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64), which is related to cars and fuel consumption of year 2014.

Examples of features:

- MODELYEAR e.g. 2014
- MAKE e.g. Acura
- MODEL e.g. ILX
- VEHICLE CLASS e.g. SUV
- ENGINE SIZE e.g. 4.7
- CYLINDERS e.g 6
- TRANSMISSION e.g. A6
- FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
- FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
- FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
- CO2 EMISSIONS (g/km) e.g. 182   --> low --> 0

The question is how these various features impact CO2 emission (target)?   

To answer this question:  
1)first the csv file is read and necessary cleaning is applied.   
2)The exploratory data analysis is used to explore different features' correlation with CO2 emission  
3)both simple and multiple linear regression are applied and the accuracies are determined  
4)a GUI is designed (via tkinter) to ask the user about features of their car and based on the multiple regression model, it can predict the CO2 that their car emits.  
