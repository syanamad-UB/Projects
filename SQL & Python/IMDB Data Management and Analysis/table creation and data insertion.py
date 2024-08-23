import pandas as pd
import sqlite3
from sqlite3 import Error

def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql, drop_table_name=None):
    
    if drop_table_name: # You can optionally pass drop_table_name to drop the table. 
        try:
            c = conn.cursor()
            c.execute("""DROP TABLE IF EXISTS %s""" % (drop_table_name))
        except Error as e:
            print(e)
    
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
        
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)

    rows = cur.fetchall()

    return rows
conn_imdb = create_connection('IMDBTest.db')

def create_table_category():
    create_table_sql = """CREATE TABLE [Category](
        [CategoryID] INTEGER NOT NULL PRIMARY KEY,
        [Category] TEXT NOT NULL);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='Category')
    
def create_table_contenttype(): ##titlebasics - titletype
    create_table_sql = """CREATE TABLE [ContentType](
        [ContentTypeID] INTEGER NOT NULL PRIMARY KEY,
        [ContentType] TEXT NOT NULL);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='ContentType')
    
def create_table_genre():
    create_table_sql = """CREATE TABLE [Genre](
        [GenreID] INTEGER NOT NULL PRIMARY KEY,
        [Genre] TEXT NOT NULL);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='Genre')
    
def create_table_language():
    create_table_sql = """CREATE TABLE [Language](
        [LanguageID] INTEGER NOT NULL PRIMARY KEY,
        [Language] TEXT NOT NULL);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='Language')
    
def create_table_region():
    create_table_sql = """CREATE TABLE [Region](
        [RegionID] INTEGER NOT NULL PRIMARY KEY,
        [Region] TEXT NOT NULL);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='Region')
    
def create_table_role():
    create_table_sql = """CREATE TABLE [Role](
        [RoleID] INTEGER NOT NULL PRIMARY KEY,
        [Role] TEXT NOT NULL);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='Role')
    
def create_table_titletype(): ##titleakas - types
    create_table_sql = """CREATE TABLE [TitleType](
        [TitleTypeID] INTEGER NOT NULL PRIMARY KEY,
        [TitleType] TEXT NOT NULL);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='TitleType')
    
def create_table_title(): ## titlebasics
    create_table_sql = """CREATE TABLE [Title](
        [TitleID] TEXT NOT NULL PRIMARY KEY,
        [ContentTypeID] INTEGER NOT NULL,
        [PrimaryTitle] TEXT NOT NULL,
        [OriginalTitle] TEXT,
        [IsAdult] INTEGER,
        [StartYear] INTEGER,
        [EndYear] INTEGER,
        [RunTimeMinutes] INTEGER,
        FOREIGN KEY(ContentTypeID) REFERENCES ContentType(ContentTypeID));"""
    create_table(conn_imdb, create_table_sql,drop_table_name='Title')
    
def create_table_person(): ## name
    create_table_sql = """CREATE TABLE [Person](
        [PersonID] TEXT NOT NULL PRIMARY KEY,
        [PersonName] TEXT NOT NULL,
        [BirthYear] INTEGER,
        [DeathYear] INTEGER);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='Person')
    
def create_table_personrole():
    create_table_sql = """CREATE TABLE [PersonRole](
        [PersonID] TEXT NOT NULL,
        [RoleID] INTEGER NOT NULL,
        [Order] INTEGER NOT NULL,
        PRIMARY KEY(PersonID,RoleID),
        FOREIGN KEY(PersonID) REFERENCES Person(PersonID),
        FOREIGN KEY(RoleID) REFERENCES Role(RoleID));"""
    create_table(conn_imdb, create_table_sql,drop_table_name='PersonRole')
    
def create_table_titleakas():
    create_table_sql = """CREATE TABLE [TitleAkas](
        [TitleID] TEXT,
        [Order] INTEGER NOT NULL,
        [LocalisedTitle] TEXT NOT NULL,
        [RegionID] INTEGER,
        [LanguageID] INTEGER,
        [Attributes] TEXT,
        [IsOriginalTitle] INTEGER,
        PRIMARY KEY([TitleID],[Order]),
        FOREIGN KEY(TitleID) REFERENCES Title(TitleID),
        FOREIGN KEY(RegionID) REFERENCES Region(RegionID),
        FOREIGN KEY(LanguageID) REFERENCES Language(LanguageID));"""
    create_table(conn_imdb, create_table_sql,drop_table_name='TitleAkas')
    
def create_table_titlegenre():
    create_table_sql = """CREATE TABLE [TitleGenre](
        [TitleID] TEXT,
        [GenreID] INTEGER,
        [Order] INTEGER NOT NULL,
        PRIMARY KEY(TitleID,GenreID),
        FOREIGN KEY(TitleID) REFERENCES Title(TitleID),
        FOREIGN KEY(GenreID) REFERENCES Genre(GenreID));"""
    create_table(conn_imdb, create_table_sql,drop_table_name='TitleGenre')
    
def create_table_titleprincipal():
    create_table_sql = """CREATE TABLE [TitlePrincipal](
        [TitleID] TEXT,
        [PersonID] TEXT,
        [Order] INTEGER NOT NULL,
        [CategoryID] INTEGER NOT NULL,
        [Job] TEXT,
        [RoleNames] TEXT,
        PRIMARY KEY([TitleID],[PersonID],[Order]),
        FOREIGN KEY(TitleID) REFERENCES Title(TitleID),
        FOREIGN KEY(PersonID) REFERENCES Person(PersonID),
        FOREIGN KEY(CategoryID) REFERENCES Category(CategoryID));"""
    create_table(conn_imdb, create_table_sql,drop_table_name='TitlePrincipal')
    
def create_table_titleratings():
    create_table_sql = """CREATE TABLE [TitleRatings](
        [TitleID] TEXT NOT NULL PRIMARY KEY,
        [AverageRating] REAL NOT NULL,
        [NumVotes] INTEGER NOT NULL);"""
    create_table(conn_imdb, create_table_sql,drop_table_name='TitleRatings')

create_table_category()
create_table_contenttype()
create_table_genre()
create_table_language()
create_table_region()
create_table_role()
create_table_titletype()
create_table_title()
create_table_person()
create_table_personrole()
create_table_titleakas()
create_table_titlegenre()
create_table_titleprincipal()
create_table_titleratings()


with open('titleprincipals.tsv', 'r') as file:
    header = None
    titleprincipals = []
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        # do something with line
        if not header:
            header = line.strip().split('\t')
            continue
        l = line.strip().split('\t')
        l = ['' if x == '\\N' else x for x in l]
        l = [x.replace('"','') for x in l]
        titleprincipals.append(l)
with open('titlebasics.tsv', 'r') as file:
    header = None
    titlebasics = []
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        # do something with line
        if not header:
            header = line.strip().split('\t')
            continue
        l = line.strip().split('\t')
        l = ['' if x == '\\N' else x for x in l]
        l[8] = l[8].split(',')
        titlebasics.append(l)
with open('titleakas.tsv', 'r') as file:
    header = None
    titleakas = []
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        # do something with line
        if not header:
            header = line.strip().split('\t')
            continue
        l = line.strip().split('\t')
        l = ['' if x == '\\N' else x for x in l]
        titleakas.append(l)
with open('name.tsv', 'r') as file:
    header = None
    name = []
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        # do something with line
        if not header:
            header = line.strip().split('\t')
            continue
        l = line.strip().split('\t')
        l = ['' if x == '\\N' else x for x in l]
        l[4] = l[4].split(',')
        l[5] = l[5].split(',')
        name.append(l)
with open('titleratings.tsv', 'r') as file:
    header = None
    titleratings = []
    for line in file:
        if not line.strip(): # used for skipping empty lines!
            continue
        # do something with line
        if not header:
            header = line.strip().split('\t')
            continue
        l = line.strip().split('\t')
        titleratings.append(l)
categories = sorted(list(set([(i[3],) for i in titleprincipals])))
def insert_table_category():
    sql = ''' INSERT INTO Category(Category)
                VALUES(?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,categories)
    conn_imdb.commit()
    cur.close()
insert_table_category()
contenttype = sorted(list(set([(i[1],) for i in titlebasics])))
def insert_table_contenttype():
    sql = ''' INSERT INTO ContentType(ContentType)
                VALUES(?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,contenttype)
    conn_imdb.commit()
    cur.close()
insert_table_contenttype()
genre = [i[8] for i in titlebasics]
genre =  [j for i in genre for j in i]
genre = sorted(list(set(genre)))
genre.remove('')
genre = [(i,) for i in genre]
def insert_table_genre():
    sql = ''' INSERT INTO Genre(Genre)
                VALUES(?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,genre)
    conn_imdb.commit()
    cur.close()
insert_table_genre()
language = sorted(list(set([i[4] for i in titleakas])))
language.remove('')
language = [(i,) for i in language]
def insert_table_language():
    sql = ''' INSERT INTO Language(Language)
                VALUES(?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,language)
    conn_imdb.commit()
    cur.close()
insert_table_language()
region = sorted(list(set([i[3] for i in titleakas])))
region.remove('')
region = [(i,) for i in region]
def insert_table_region():
    sql = ''' INSERT INTO Region(Region)
                VALUES(?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,region)
    conn_imdb.commit()
    cur.close()
insert_table_region()
role = [i[4] for i in name]
role =  [j for i in role for j in i]
role = sorted(list(set(role)))
role.remove('')
role = [(i,) for i in role]
def insert_table_role():
    sql = ''' INSERT INTO Role(Role)
                VALUES(?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,role)
    conn_imdb.commit()
    cur.close()
insert_table_role()
titletype = list(set([i[5] for i in titleakas]))
titletype = [i.replace('\x02',',').split(',') for i in titletype]
titletype =  sorted(list(set([j for i in titletype for j in i])))
titletype.remove('')
titletype = [(i,) for i in titletype]
def insert_table_titletype():
    sql = ''' INSERT INTO TitleType(TitleType)
                VALUES(?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,titletype)
    conn_imdb.commit()
    cur.close()
insert_table_titletype()
sql_statement = "select ContentTypeID,ContentType from ContentType;"
contenttypes = execute_sql_statement(sql_statement, conn_imdb)
k = {}
for i in contenttypes:
    k[i[1]] = i[0]
titles = [(j[0],k[j[1]],j[2],j[3],j[4],j[5],j[6],j[7]) for j in titlebasics]
def insert_table_title():
    sql = ''' INSERT INTO Title(TitleID,ContentTypeID,PrimaryTitle,OriginalTitle,IsAdult,StartYear,EndYear,RunTimeMinutes)
                VALUES(?,?,?,?,?,?,?,?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,titles)
    conn_imdb.commit()
    cur.close()
insert_table_title()
persons = [(i[0],i[1],i[2],i[3]) for i in name]
def insert_table_person():
    sql = ''' INSERT INTO Person(PersonID,PersonName,BirthYear,DeathYear)
                VALUES(?,?,?,?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,persons)
    conn_imdb.commit()
    cur.close()
insert_table_person()
sql_statement = "select RoleID,Role from Role;"
roles = execute_sql_statement(sql_statement, conn_imdb)
k = {}
for i in roles:
    k[i[1]] = i[0]
l = []
for i in name:
    l = l + [(i[0],z[0],z[1]) for z in list(map(list,zip(*[[k[j] for j in i[4]],list(range(1,len(i[4])))])))]
def insert_table_personrole():
    sql = ''' INSERT INTO PersonRole(PersonID,RoleID,Order)
                VALUES(?,?,?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,l)
    conn_imdb.commit()
    cur.close()
insert_table_personrole()
sql_statement = "select RegionID,Region from Region;"
regions = execute_sql_statement(sql_statement, conn_imdb)
R = {}
for i in regions:
    R[i[1]] = i[0]
sql_statement = "select LanguageID,Language from Language;"
languages = execute_sql_statement(sql_statement, conn_imdb)
L = {}
for i in languages:
    L[i[1]] = i[0]
titleaka = [(i[0],i[1],i[2],R[i[3]] if i[3] != '' else 0,L[i[3]] if i[4] != '' else 0,i[6],i[7]) for i in titleakas]
def insert_table_titleakas():
    sql = ''' INSERT INTO TitleAkas(TitleID,Order,LocalisedTitle,RegionID,LanguageID,Attributes,IsOriginalTitle)
                VALUES(?,?,?,?,?,?,?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,titleaka)
    conn_imdb.commit()
    cur.close()
insert_table_titleakas()
sql_statement = "select GenreID,Genre from Genre;"
genres = execute_sql_statement(sql_statement, conn_imdb)
k = {}
for i in genres:
    k[i[1]] = i[0]
l = []
for i in titlebasics:
    l = l + [(i[0],z[0],z[1]) for z in list(map(list,zip(*[[k[j] for j in i[8]],list(range(1,len(i[8])))])))]
def insert_table_titlegenre():
    sql = ''' INSERT INTO TitleGenre(TitleID,GenreID,Order)
                VALUES(?,?,?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,l)
    conn_imdb.commit()
    cur.close()
insert_table_titlegenre()
sql_statement = "select CategoryID,Category from Category;"
category = execute_sql_statement(sql_statement, conn_imdb)
k = {}
for i in category:
    k[i[1]] = i[0]
titleprincipal = [(i(0),i(2),i(1),k[i(3)],i[4],i[5]) for i in titleprincipals]
def insert_table_titleprincipal():
    sql = ''' INSERT INTO TitlePrincipal(TitleID,PersonID,Order,CategoryID,Job,RoleNames)
                VALUES(?,?,?,?,?,?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,titleprincipal)
    conn_imdb.commit()
    cur.close()
insert_table_titleprincipal()

titlerating = [tuple(i) for i in titleratings]
def insert_table_titleratings():
    sql = ''' INSERT INTO TitleRatings(TitleID,AverageRating,NumVotes)
                VALUES(?,?,?) '''  
    cur = conn_imdb.cursor()
    cur.executemany(sql,titlerating)
    conn_imdb.commit()
    cur.close()
insert_table_titleratings()