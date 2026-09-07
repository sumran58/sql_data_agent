import psycopg2

class DatabaseUtil:
    def __init__(self,db_config):
        self.db_config=db_config
        try:
            self.connection=psycopg2.connect(**db_config) # **unpaks a dictionary and * unpacks the list or tuple
        except Exception as e:
            print(f"Error connecting to the database {e}")
            self.connection=None

    def schema_details(self,schema_name):
        try:
            schema_info_context=""
            connection=self.connection
            cursor=connection.cursor() #cursor is used to execute the query
            schema_info_context=f"Database schema context :{schema_name}\n" #schema is a folder name inside a database and by default it is public

            cursor.execute("select table_name from information_schema.tables where table_schema=%s;",(schema_name,))
            tables_list=cursor.fetchall()

            for table in tables_list:
                table_name=table[0]
                schema_info_context=f"{schema_info_context}\nTable: {table_name}\n"

                cursor.execute("select column_name,data_type from information_schema.columns where table_name=%s;",(table_name,))
                column_list=cursor.fetchall()

                for column in column_list:
                    column_name=column[0]
                    data_type=column[1]
                    schema_info_context=f"{schema_info_context}\n Column:{column_name} , Data Type: {data_type}"
        
                cursor.execute(f"select * from {schema_name}.{table_name} limit 5")
                sample_data=cursor.fetchall()
                for row in sample_data:
                    schema_info_context=f"{schema_info_context}   {row}\n"
        except Exception as e:
            print(f"error fetching the schema details {e}")
            schema_info_context=f"error fetching the schema details :{e}"
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
        return schema_info_context

    def execute_sql(self, query):
        try:
            connection = self.connection
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            return str(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    



obj = DatabaseUtil({
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "sim123",
    "dbname": "postgres"
})

result = obj.schema_details("public")

with open("test_schema_details.txt", "w") as f:
    f.write(result)
            
