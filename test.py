import pandas as pd
df = pd.read_csv('breast-cancer.csv')
df.columns = [c.lower() for c in df.columns]

from sqlalchemy import create_engine
engine = create_engine('postgresql://﻿postgres:Password990@localhost:5432/ML1')

df.to_sql("breast_cancer", engine)

