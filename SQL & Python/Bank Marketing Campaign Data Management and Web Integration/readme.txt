The Bank Marketing Data for this project is taken from UCI Machine Learning Repository and below is the link to the data source: https://archive.ics.uci.edu/ml/datasets/bank+marketing

This dataset pertains to the direct marketing campaigns of a Portuguese banking institution and can be used to predict whether a potential customer is likely to subscribe to a term deposit or not .

Columns which are not necessary for the analysis are removed and the data is split into multiple tables based on the attribute types mentioned in the data source link. Fake data is generated for few columns like first_name, last_name, city_residence….

The resultant data files are stored in csv format and are located in Milestone2_Group31_ProjectFiles/Data folder.

The database ‘bankmarketing’ is created and then tables are created in the database using create.sql which is located in Milestone2_Group31_ProjectFiles/Scripts folder.

Once the tables are created, the data is loaded to the database using python scripts. We used psycopg2 library to connect to bankmarketing database. The csv files are read using pandas library and are loaded to the tables through the connection established to the database using psycopg2 library. The script (load.ipynb) is located in Milestone2_Group31_ProjectFiles/Scripts folder.

Once the data is loaded, it is tested using multiple sql queries which include data insert, update, delete, select and trigger queries. Optimization is done by creating indexes. The scripts for these are included in the report.

Bonus website files are located in Milestone2_Group31_ProjectFiles/Website folder
