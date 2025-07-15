#%%
import springernature_api_client.openaccess as openaccess
import sqlite3
import pandas as pd
import time
import urllib.parse
#%%
con = sqlite3.connect('papers.db')
cursor = con.cursor()
key = '4ed70543f167bb7de765cc2419759d20'

# %% Connect to openaccessAPI with api_key
openaccess_client = openaccess.OpenAccessAPI(api_key=key)

# %% try different queriers, in order to get the most related papers possible
query_list =  ["drug discovery", 'AI in drug discovery', 'Artificial intelligence in drug discovery', 
               'Drug Discovery', 'AI in Drug Discovery', 'Artificial intelligence in Drug Discovery']
all_records = []
# iteratively request bacthes of 20 records across 50 pages
all_records = []  # One big list to hold all articles

for query in query_list:
    for i in range(10):
        start = 6621 + i * 20 
        response = openaccess_client.search(
            q=query,
            p=20,
            s=start,
            fetch_all=False,
            is_premium=False
        )

        # If response is a string, convert to dict
        if isinstance(response, str):
            import json
            response = json.loads(response)

        # Append all records from this page
        if isinstance(response, dict) and "records" in response:
            all_records.extend(response["records"])
            print(f"Fetched {len(response['records'])} records from query='{query}' page={i+1}")
        else:
            print(f"No records for query='{query}' at page={i+1}")

 #%% flatten result
records = []
for record in all_records:    
    flat_record = {
        'title': record['title'],
        'doi': record['doi'],
        'publicationName': record['publicationName'],
        'publicationDate': record['publicationDate'],
        'publisher': record['publisher'],
        'abstract': record["abstract"]["p"] if isinstance(record.get("abstract"), dict)
                else record.get("abstract", ""),
        'language': record['language'],
        'creators': [c.get('creator', '') for c in record.get('creators', [])],
        'subjects': record.get('subjects', []),
        'disciplines': [d['term'] for d in record.get('disciplines', [])],
        'url': record['url'][0]['value'] if record['url'] else None
    }
    records.append(flat_record)
    #%% Store flattened data into dataframe and save in sql table
    df = pd.DataFrame(records)
    #%%
    for col in df.columns:
            df[col]=df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df.drop_duplicates(subset="doi", inplace=True)
    
    df.to_sql("paper", con, if_exists="append", index=False)
    con.close()

# %%
