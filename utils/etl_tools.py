import os
import requests
import pandas as pd
class ETLTools:
    def __init__(self):
        pass

    def extract_load(self,url:str,output_folder:str,format:str):
        """
        this tool extract the data from the api(url) and loads it into the destination (folder)
        
        args:
        url(str):the api endpoint from whic the data is extracted 
        output_folder(str):the destination where the extracted data is stored
        
        returns:
         the message indicating the success or the failure of the operation 

        """
        project_root=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
        output_folder=os.path.join(project_root,output_folder)

        try:
            response=requests.get(url)
            response.raise_for_status()
            data=response.json()
            filename=os.path.join(output_folder,f"extracted data .{format}")
            os.makedirs(output_folder,exist_ok=True)
            df=pd.json_normalize(data["results"])
            if format=='csv':
                df.to_csv(filename,index=False)
            elif format=='json':
                df.to_json(filename,orients="records",lines=True)
            elif format=='parquet':
                df.to_parquet(filename,index=False)
            else:
                return f"unsupported format {format}"
            return f"data successfully extracted and saved to {filename}"
        except requests.exceptions.RequestException as e:
            return f"failed to extract the data: {e}"

    def transform_load_context(self, file_path:str):
        """
        This tool transforms the data from the specified file and loads it into the
        desired location (output_folder).

        Args:
            file_path (str): The path to the file containing the data to be transformed.
            output_folder (str): The folder where the transformed data will be saved.
            output_format (str): The format in which to save the transformed data (csv, json, parquet).
        Returns:
            str: A message indicating the success or failure of the operation.
        """

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".json":
            df = pd.read_json(file_path, lines=True)
        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_extension}"

        top_3_rows = str(df.head(3))

        return top_3_rows

    def execute_code(self,code:str):
        """
        This tool executes the provided code and returns the output.

        Args:
            code (str): The code to be executed.
        Returns:
            str: The output of the executed code or an error message if execution fails.
        """

        try:
            exec(code)
            return "Code executed successfully."
        except Exception as e:
            return f"Failed to execute code: {e}"



if __name__=="__main__":
    obj=ETLTools()
    path="data//extract//extracted data .csv"
    print(obj.transform_load_context(path))