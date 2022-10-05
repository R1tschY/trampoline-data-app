# imports
import pandas as pd
import os
from more_itertools import strip
import sqlite3
import logging
from sqlalchemy import create_engine
import pymysql


logging.basicConfig(level=logging.DEBUG)

# PDF processing

df_data = pd.read_csv('data.csv') # data.csv contains information about events and their resultbooks
header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified", "Event", "Phase", "Year", "Location", "Gender")
df_main = pd.DataFrame(columns=header) # main dataframe
df_mainpos = 0
for idx, row in df_data.iterrows():

    event_phase = row['Event Phase'] # qualification, semi, final
    association = row['Association'] # europe, world

    location = row['Location']
    year = row['Year']
    event = row['Event']

    if "Women" in row['Output']:
        gender = "Women"
    elif "Men" in row['Output']:
        gender = "Men"
    else:
        gender = "Unknown"

    first_page = str(row['First'])
    last_page = str(row['Last'])
    output_name = 'lists/' + row['Output']
    input_name = '"' + row['Input'] + '"'
    logging.info('Event: {} {}, Association: {}, Phase: {}, Gender: {}'.format(event, year, association, event_phase, gender))
    # use ghostscript to convert pdf to txt
    os.system('gs -sDEVICE=txtwrite -dFirstPage=' + first_page + ' -dLastPage=' + last_page + ' -o ' + output_name + '.txt ' + input_name)

    # Detect lines with information
    if association == 'world':
        if event_phase == 'Qualification':
            file1 = open(output_name + '.txt', 'r')
            Lines = file1.readlines()
            
            count = 0
            entry_counts = []

            for line in Lines:
                if "1st" in line and not "DNS" in line:
                    entry_counts.append(count)
                elif "2nd" in line and not "DNS" in line:
                    entry_counts.append(count)
                count += 1
        else:
            file1 = open(output_name + '.txt', 'r')
            Lines = file1.readlines()
            
            count = 0
            rank_count = 1
            entry_counts = []
            for line in Lines:
                stripped_line = line.strip().split()
                if len(stripped_line) > 4 and stripped_line[0] == str(rank_count):
                    entry_counts.append(count)
                    rank_count += 1
                count += 1

    elif association == 'europe':
        file1 = open(output_name + '.txt', 'r')
        Lines = file1.readlines()
        
        count = 0
        rank_count = 1
        entry_counts = []
        for line in Lines:
            stripped_line = line.strip().split()
            if stripped_line[0] == str(rank_count) and len(stripped_line) > 8:
                entry_counts.append(count)
                rank_count += 1
            count += 1
    else:
        print("Wrong association")


    # Processing information to dataframe
    # in world qualification 
    df = pd.DataFrame()
    if association == 'world':
        if event_phase == 'Qualification':
            header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified", "Event", "Phase", "Year", "Location", "Gender")
            df = pd.DataFrame(columns=header)

            df_pos = 0
            for entry in entry_counts:
                
                if df_pos%2 == 0:
                    corrector = 1
                else:
                    corrector = -1

                current_entry = Lines[entry].strip().split()
                corrected_entry = Lines[entry+corrector].strip().split()

                rank = int(corrected_entry[0])
                routine = current_entry[0]
                d = float(current_entry[1])
                e = float(current_entry[2])
                t = float(current_entry[3])
                h = float(current_entry[4])

                if len(current_entry) == 7:
                    penalty = 0.0
                    total = float(current_entry[5])
                else:
                    penalty = float(current_entry[5])
                    total = float(current_entry[6])


                if len(corrected_entry) == 6:

                    if len(corrected_entry[-1]) > 3:
                        name = corrected_entry[1] + " " + corrected_entry[2] + " " + corrected_entry[3]
                        country = corrected_entry[4]
                        end_added = float(corrected_entry[5])
                    else:
                        name = corrected_entry[1] + " " + corrected_entry[2]
                        country = corrected_entry[3]
                        end_added = float(corrected_entry[4])

                    if corrected_entry[-1].isupper():
                        qualified = corrected_entry[-1]
                    else:
                        qualified = ' '

                elif len(corrected_entry) == 7:
                    name = corrected_entry[1] + " " + corrected_entry[2] + " " + corrected_entry[3]
                    country = corrected_entry[4]
                    end_added = float(corrected_entry[5])

                    if corrected_entry[-1].isupper():
                        qualified = corrected_entry[-1]
                    else:
                        qualified = ' '

                elif len(corrected_entry) == 8:
                    name = corrected_entry[1] + " " + corrected_entry[2] + " " + corrected_entry[3] + " " + corrected_entry[4]
                    country = corrected_entry[5]
                    end_added = float(corrected_entry[6])

                    if corrected_entry[-1].isupper():
                        qualified = corrected_entry[-1]
                    else:
                        qualified = ' '
                else:
                    qualified = ""
                    name = corrected_entry[1] + " " + corrected_entry[2]
                    country = corrected_entry[3]
                    end_added = float(corrected_entry[4])

                    if corrected_entry[-1].isupper():
                        qualified = corrected_entry[-1]
                    else:
                        qualified = ' '

                df.loc[df_pos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  penalty,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                 ) 

                df_main.loc[df_mainpos] = (rank,
                                           routine,
                                           name,
                                           d, 
                                           e,
                                           t,
                                           h,
                                           penalty,
                                           total,
                                           end_added,
                                           country,
                                           qualified,
                                           event,
                                           event_phase,
                                           year,
                                           location,
                                           gender
                                          )  
                                          
                if df_pos == len(entry_counts)-3:
                    break
                df_pos += 1
                df_mainpos += 1
            df.to_csv(output_name + ".csv", index=False)
        elif event_phase == "Semi":
            header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified", "Event", "Phase", "Year", "Location", "Gender")
            df = pd.DataFrame(columns=header)

            df_pos = 0
            for entry in entry_counts:

                current_entry = Lines[entry].strip().split()
                next_entry = Lines[entry+1].strip().split()

                if current_entry[3].isupper() == False and current_entry[4].isupper() == True:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2] + ' ' + current_entry[3]
                    country = current_entry[4]
                    total = float(current_entry[5])
                    d = float(next_entry[0])
                    e = float(next_entry[1])
                    t = float(next_entry[2])
                    h = float(next_entry[3])
                    if len(next_entry) == 4:
                        pen = 0.00
                    else:
                        pen = float(next_entry[4])
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '
                elif current_entry[4].isupper() == False and len(current_entry) > 6:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2] + ' ' + current_entry[3] + ' ' + current_entry[4]
                    # print(current_entry)
                    country = current_entry[5]
                    total = float(current_entry[6])
                    d = float(next_entry[0])
                    e = float(next_entry[1])
                    t = float(next_entry[2])
                    h = float(next_entry[3])
                    if len(next_entry) == 4:
                        pen = 0.00
                    else:
                        pen = float(next_entry[4])
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '
                else:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2]
                    country = current_entry[3]
                    total = float(current_entry[4])
                    d = float(next_entry[0])
                    e = float(next_entry[1])
                    t = float(next_entry[2])
                    h = float(next_entry[3])
                    if len(next_entry) == 4:
                        pen = 0.00
                    else:
                        pen = float(next_entry[4])
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '

                routine = "1st"
                end_added = float(total)
                df.loc[df_pos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )  
                df_main.loc[df_mainpos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                ) 
                
                df_pos += 1
                df_mainpos += 1
                # header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified")
                # header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified", "Event", "Phase", "Year", "Location")
            df.to_csv(output_name + ".csv", index=False)
        elif event_phase == "Final":
            header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified", "Event", "Phase", "Year", "Location", "Gender")
            df = pd.DataFrame(columns=header)

            df_pos = 0
            for entry in entry_counts:

                current_entry = Lines[entry].strip().split()
                next_entry = Lines[entry+1].strip().split()

                if current_entry[3].isupper() == False:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2] + ' ' + current_entry[3]
                    country = current_entry[4]
                    total = float(current_entry[5])
                    d = float(next_entry[0])
                    e = float(next_entry[1])
                    t = float(next_entry[2])
                    h = float(next_entry[3])
                    if len(next_entry) == 4:
                        pen = 0.00
                    else:
                        pen = float(next_entry[4])
                elif current_entry[4].isupper() == False and len(current_entry) > 5:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2] + ' ' + current_entry[3] + ' ' + current_entry[4]
                    country = current_entry[5]
                    total = float(current_entry[6])
                    d = float(next_entry[0])
                    e = float(next_entry[1])
                    t = float(next_entry[2])
                    h = float(next_entry[3])
                    if len(next_entry) == 4:
                        pen = 0.00
                    else:
                        pen = float(next_entry[4])
                else:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2]
                    country = current_entry[3]
                    total = float(current_entry[4])
                    d = float(next_entry[0])
                    e = float(next_entry[1])
                    t = float(next_entry[2])
                    h = float(next_entry[3])
                    if len(next_entry) == 4:
                        pen = 0.00
                    else:
                        pen = float(next_entry[4])

                routine = "1st"
                qualified = ''
                end_added = float(total)
                df.loc[df_pos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )  
                df_main.loc[df_mainpos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                ) 
                
                df_pos += 1
                df_mainpos += 1
                # header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified")
            df.to_csv(output_name + ".csv", index=False)
        else:
            print('Error in Association: world event phase')

    elif association == 'europe':
        if event_phase == 'Qualification':
            header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified", "Event", "Phase", "Year", "Location", "Gender")
            df = pd.DataFrame(columns=header)

            df_pos = 0
            for entry in entry_counts:

                corrector = 1
                current_entry = Lines[entry].strip().split()
                corrected_entry = Lines[entry+corrector].strip().split()
                
                if len(current_entry) < 13:
                    rank = int(current_entry[0])
                    routine = current_entry[4]
                    name = current_entry[1] + ' ' + current_entry[2]
                    d = 0.00
                    e = 0.00
                    t = 0.00
                    h = 0.00
                    pen = 0.00
                    total = 0.00
                    end_score = float(current_entry[11])
                    country = current_entry[3]
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '
                elif current_entry[3].isupper() == False:
                    rank = int(current_entry[0])
                    routine = current_entry[5]
                    name = current_entry[1] + ' ' + current_entry[2] + ' ' + current_entry[3]
                    d = float(current_entry[8])
                    e = float(current_entry[6])
                    t = float(current_entry[7])
                    h = float(current_entry[9])
                    pen = float(current_entry[12])
                    total = float(current_entry[11])
                    end_score = float(current_entry[13])
                    country = current_entry[4]
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '
                else:
                    rank = int(current_entry[0])
                    routine = current_entry[4]
                    name = current_entry[1] + ' ' + current_entry[2]
                    d = float(current_entry[7])
                    e = float(current_entry[5])
                    t = float(current_entry[6])
                    h = float(current_entry[8])
                    pen = float(current_entry[11])
                    total = float(current_entry[10])
                    end_score = float(current_entry[12])
                    country = current_entry[3]
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '

                if routine == '1' or routine == 1:
                    routine = '1st'
                # if name == 'ROMERO ROSARIO Noemi':
                #     print(routine)
                #     print(type(routine))
                #     print(len(routine))
                end_added = float(end_score)
                df.loc[df_pos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )  
                df_main.loc[df_mainpos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )  
                
                df_pos += 1
                df_mainpos += 1
                if len(corrected_entry) < 8:
                    routine = corrected_entry[0]
                    d = 0.00
                    e = 0.00
                    t = 0.00
                    h = 0.00
                    pen = 0.00
                    total = 0.00
                    end_score = float(current_entry[11])
                elif len(current_entry) < 13:
                    routine = corrected_entry[0]
                    d = float(corrected_entry[3])
                    e = float(corrected_entry[1])
                    t = float(corrected_entry[2])
                    h = float(corrected_entry[4])
                    pen = float(corrected_entry[7])
                    total = float(corrected_entry[6])
                    end_score = float(current_entry[11])
                else:
                    routine = corrected_entry[0]
                    d = float(corrected_entry[3])
                    e = float(corrected_entry[1])
                    t = float(corrected_entry[2])
                    h = float(corrected_entry[4])
                    pen = float(corrected_entry[7])
                    total = float(corrected_entry[6])
                    end_score = float(current_entry[12])

                if routine == '2' or routine == 2:
                    routine = '2nd'

                df.loc[df_pos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )  

                df_main.loc[df_mainpos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )  
                
                df_pos += 1
                df_mainpos += 1
                # header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified")
            df.to_csv(output_name + ".csv", index=False)
        elif event_phase == 'Semi':
            header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified", "Event", "Phase", "Year", "Location", "Gender")
            df = pd.DataFrame(columns=header)

            df_pos = 0
            for entry in entry_counts:

                current_entry = Lines[entry].strip().split()

                if len(current_entry) < 11:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2]
                    d = 0.00
                    e = 0.00
                    t = 0.00
                    h = 0.00
                    pen = 0.00
                    total = 0.00
                    country = current_entry[3]
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '
                elif current_entry[3].isupper() == False:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2] + ' ' + current_entry[3]
                    d = float(current_entry[8])
                    e = float(current_entry[6])
                    t = float(current_entry[7])
                    h = float(current_entry[9])
                    pen = float(current_entry[10])
                    total = float(current_entry[11])
                    country = current_entry[4]
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '
                else:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2]
                    d = float(current_entry[7])
                    e = float(current_entry[5])
                    t = float(current_entry[6])
                    h = float(current_entry[8])
                    pen = float(current_entry[9])
                    total = float(current_entry[10])
                    country = current_entry[3]
                    if current_entry[-1].isupper():
                        qualified = current_entry[-1]
                    else:
                        qualified = ' '


                routine = "1st"
                end_added = float(total)
                df.loc[df_pos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )  

                df_main.loc[df_mainpos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )  
                
                df_pos += 1
                df_mainpos += 1
                # header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified")
            df.to_csv(output_name + ".csv", index=False)
        elif event_phase == 'Final':
            header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified", "Event", "Phase", "Year", "Location", "Gender")
            df = pd.DataFrame(columns=header)

            df_pos = 0
            for entry in entry_counts:

                corrector = 1
                current_entry = Lines[entry].strip().split()
                corrected_entry = Lines[entry+corrector].strip().split()

                if len(current_entry) < 11:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2]
                    d = 0.00
                    e = 0.00
                    t = 0.00
                    h = 0.00
                    pen = 0.00
                    total = 0.00
                    country = current_entry[3]
                elif current_entry[3].isupper() == False:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2] + ' ' + current_entry[3]
                    d = float(current_entry[8])
                    e = float(current_entry[6])
                    t = float(current_entry[7])
                    h = float(current_entry[9])
                    pen = float(current_entry[10])
                    total = float(current_entry[11])
                    country = current_entry[4]
                else:
                    rank = int(current_entry[0])
                    name = current_entry[1] + ' ' + current_entry[2]
                    d = float(current_entry[7])
                    e = float(current_entry[5])
                    t = float(current_entry[6])
                    h = float(current_entry[8])
                    pen = float(current_entry[9])
                    total = float(current_entry[10])
                    country = current_entry[3]


                routine = "1st"
                end_added = float(total)
                qualified = ''
                df.loc[df_pos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )

                df_main.loc[df_mainpos] = (rank,
                                  routine,
                                  name,
                                  d, 
                                  e,
                                  t,
                                  h,
                                  pen,
                                  total,
                                  end_added,
                                  country,
                                  qualified,
                                  event,
                                  event_phase,
                                  year,
                                  location,
                                  gender
                                )      
                
                df_pos += 1
                df_mainpos += 1
            df.to_csv(output_name + ".csv", index=False)
                # header = ("Rank", "Routine", "Name", "D", "E", "T", "H", "Pen.", "Total", "End", "Country", "Qualified")

# connection = sqlite3.connect("trampoline.db")
# cursor = connection.cursor()
# df_main.to_sql(name="ranklists", con=connection, if_exists='replace')
# connection.close()
engine = create_engine('mysql+pymysql://falkoin:Tim!nA|$tr0wd$10wSp3rSEs@localhost/trampoline', pool_recycle=3600)
with engine.connect() as conn:
    df_main.to_sql(name="ranklists", con=conn, if_exists='replace')