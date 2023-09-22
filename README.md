```
Install the dependencies using requirements.txt

```
pip install -r requirements.txt
```

Add your OpenAI Key by creating a .env file in the folder and add the following within it:
```
OPENAI_API_KEY="<your key>"
```
If you who would like to use the HuggingFace Approach, be sure to add the HuggingFace API Key in your .env file:
```
HUGGINGFACEHUB_API_TOKEN="<your key>"
```

Run the App
```
streamlit run app.py
```

**NOTE:** Please keep in mind that you need to check the hardware requirements for the model you choose based on your machine,
as the embeddings and the model will run locally on your system, and will be loaded in your RAM. Be sure to do some research before running the code with any choosen model.




