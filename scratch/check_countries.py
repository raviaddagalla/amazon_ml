import sys
sys.path.insert(0, 'd:/Nandhu/amazon_ml/student_resource/code/business_entity_resolution')
from src.config import TRAIN_S1
import duckdb

con = duckdb.connect()
print("Distinct countries:")
res = con.execute(f"SELECT DISTINCT country, count(*) FROM read_csv('{TRAIN_S1}', delim='\\t', header=true, quote='', escape='') GROUP BY country").fetchall()
print(res)
