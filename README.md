# moth
chatbot api 

#installation instructions
-make a virtualenv with python 3
-pip install -r requirements.txt
-for each project, make a keys/openai_XXX.txt file, where the contents of the file are the API key to openai, and XXX is replaced by the name of the project
-for each project, populate the data/XXX directory with the raw (pre-embedded) knowledge base date
-for each project, populate the embeddings/XXX directory with the knowledge base embeddings, so two serialized numpy arrays


