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



if __name__=="__main__":
    obj=ETLTools()
    print(obj.extract_load("https://pokeapi.co/api/v2/pokemon/","data/extract","csv"))