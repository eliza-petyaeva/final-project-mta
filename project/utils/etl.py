import pandas as pd
import base64

def download_csv(df, filename) -> str:
    # df = pd.DataFrame(dict_).reset_index()
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href
