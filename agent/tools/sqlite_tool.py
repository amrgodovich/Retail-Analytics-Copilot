import sqlite3

class SQLiteTool:
    def __init__(self, db_path="data/northwind.sqlite"):
        # for opening connection
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def run_sql(self, sql):
        try:
            cur = self.conn.cursor()
            cur.execute(sql)

            if cur.description:
                rows = [dict(row) for row in cur.fetchall()]
            else:
                rows = []

            return True, rows, ""

        except Exception as e:
            return False, [], str(e)

    def get_tables(self):
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cur.fetchall()]

    def get_schema(self,table=None):
        schema = {}
        # if not table:
        for table in self.get_tables():
            cur = self.conn.cursor()
            cur.execute(f"PRAGMA table_info('{table}')")
            cols = []
            for row in cur.fetchall():
                cols.append({
                    "name": row[1],
                    "type": row[2]
                })
            schema[table] = cols
        # else:
        #     cur = self.conn.cursor()
        #     cur.execute(f"PRAGMA table_info('{table}')")
        #     cols = []
        #     for row in cur.fetchall():
        #         cols.append({
        #             "name": row[1],
        #             "type": row[2]
        #         })
        #     schema[table] = cols
        return schema
    

# mytest=SQLiteTool()
# # print(mytest.get_schema())
# success, rows, error = mytest.run_sql("SELECT * FROM Orders LIMIT 1;")
# print("Success:", success) 
# print("Rows:", rows)
# print("Error:", error)