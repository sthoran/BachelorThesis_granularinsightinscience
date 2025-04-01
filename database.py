#%%
import springernature_api_client.openaccess as openaccess
import sqlite3
import pandas as pd

# %%
openaccess_client = openaccess.OpenAccessAPI(api_key="4ed70543f167bb7de765cc2419759d20")
# %%
response = openaccess_client.search(q="AI in drug discovery", p=20, s=1, fetch_all=False, is_premium=False)
# %%
# Flatten each record
records = []
for record in response['records']:    
    flat_record = {
        'title': record['title'],
        'doi': record['doi'],
        'publicationName': record['publicationName'],
        'publicationDate': record['publicationDate'],
        'publisher': record['publisher'],
        'abstract': record['abstract'].get('p', []),
        'language': record['language'],
        'creators': [c['creator'] for c in record['creators']],
        'subjects': record.get('subjects', []),
        'disciplines': [d['term'] for d in record.get('disciplines', [])],
        'url': record['url'][0]['value'] if record['url'] else None
    }
    records.append(flat_record)

# %%
df = pd.DataFrame(records)
# %%
df['abstract'][0]
# %%
df
# %%
con = sqlite3.connect('papers.db')
# %%
