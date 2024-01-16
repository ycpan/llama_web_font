from clickhouse_sqlalchemy import make_session
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd

conf = {
    "user": "default",
    "password": "",
    "server_host": "10.0.0.25",
    "port": "8123",
    "db": "algorithm"
}

connection = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
engine = create_engine(connection, pool_size=100, pool_recycle=3600, pool_timeout=20)

sql = text('SHOW TABLES')

session = make_session(engine)
cursor = session.execute(sql)
try:
    fields = cursor._metadata.keys
    df = pd.DataFrame([dict(zip(fields, item)) for item in cursor.fetchall()])
    print(df)
finally:
    cursor.close()
    session.close()
