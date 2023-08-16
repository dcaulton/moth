import os
from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

class KbotController():
    def __init__(self, project):
        self.emb_model = None
        self.df_knowledge = {}
        self.project = project
        if self.project == 'triboo':
            self.kb_path = "data/triboo/Triboo_knowledgebase.csv"
            self.emb_title_path = 'embeddings/triboo/embeddings_title.npy'
            self.emb_content_path = 'embeddings/triboo/embeddings_Content.npy'
        elif self.project == 'qelp':
            self.kb_path = "data/qelp/dataset_qelp.csv"
            self.emb_title_path = 'embeddings/qelp/embeddings_title.npy'
            self.emb_content_path = 'embeddings/qelp/embeddings_Content.npy'
        else:
            raise Exception('cannot find knowledgebase files for project ' + self.project)

#    def call_all(self, query_string):
#        self.df_knowledge = self.get_knowledge_base_data()
#        self.get_embeddings_model()
#        embeddings_title, embeddings_Content = self.get_embeddings_title_and_content()
#        the_resp = self.K_BOT(query_string, embeddings_title, embeddings_Content)
#        return the_resp
#
    def knowledgebase_has_changed(self):
        klm_fullpath = self.get_last_modified_filename(self.kb_path, self.project)
        if os.path.isfile(klm_fullpath):
            return False
        else:
            return True

    def get_knowledge_base_data(self):
        df_knowledge = pd.read_csv(self.kb_path)
        df_knowledge = df_knowledge.fillna('none')
        df_knowledge.dropna(inplace=True)
        df_knowledge.reset_index(level=0, inplace=True)
        return df_knowledge

    def get_last_modified_filename(self, kb_path, project):
        klm = int(os.path.getmtime(kb_path))
        klm_filename = 'last_modified_' + str(klm)
        klm_fullpath = os.path.join('embeddings', project, klm_filename)
        return klm_fullpath

    def get_embeddings_title_and_content(self):
        embeddings_title = None
        embeddings_Content = None
        if self.knowledgebase_has_changed():
            print("knowledgebase has changed, updating embeddings")
            #Create embeddings for each column we want to compare our text with
            embeddings_title = self.build_embedding_list(self.df_knowledge['title'])
            embeddings_Content = self.build_embedding_list(self.df_knowledge['Content'])
            np.save(self.emb_title_path, np.array(embeddings_title))
            np.save(self.emb_content_path, np.array(embeddings_Content))
            klm_filename = self.get_last_modified_filename(self.kb_path, self.project)
            Path(klm_filename).touch()
        else:
            embeddings_title = np.load(self.emb_title_path, allow_pickle= True).tolist()
            embeddings_Content = np.load(self.emb_content_path, allow_pickle= True).tolist()
        return embeddings_title, embeddings_Content

    def get_embeddings_model(self):
#        emb_model=SentenceTransformer(
#            "all-mpnet-base-v2"
#        )
        self.emb_model = SentenceTransformer("./models/embedding_model")

    # Function to extract BERT embeddings for text as a list
    def calc_embeddings(self, some_text):
        if not self.emb_model:
            self.get_embeddings_model()
        text_embeddings = self.emb_model.encode(some_text,normalize_embeddings=True)
        return text_embeddings.tolist()
    # calc_embeddings('Sitel Group is changing from using the Duo App on your smart phone')

    # Function to create embeddings for each item in a list (row of a df column)
    def build_embedding_list(self, df_column):
        column_embeddings_list = list(map(self.calc_embeddings, df_column))
        return column_embeddings_list

    # Calculate CosSim between question embeddings and article embeddings
    def cos_sim_list(self, embedding_question, embedding_list):
        list_cos_sim = []
        for i in embedding_list:
            sim_pair = util.cos_sim(embedding_question,i).numpy()
            list_cos_sim.append(sim_pair[0][0])
        return list_cos_sim

    #Calculate outliers within cos_sim_max data set, identified as possible answers
    def find_outliers_IQR(self, cos_sim_max):
       q1=cos_sim_max.quantile(0.25)
       q3=cos_sim_max.quantile(0.75)
       IQR=q3-q1
       outliers = cos_sim_max[((cos_sim_max>(q3+1.5*IQR)))]

       return outliers

    def K_BOT(self, input_question, embeddings_title, embeddings_Content):
        if not self.df_knowledge:
            self.df_knowledge = self.get_knowledge_base_data()

        pd.set_option('display.max_colwidth', 5000)

        #question embeddings
        embeddings_q = self.calc_embeddings(input_question)

        #calculate cosSim for included fields
        cos_sim_max = list(map(max, self.cos_sim_list(embeddings_q, embeddings_title),
                                    self.cos_sim_list(embeddings_q, embeddings_Content)))
        self.df_knowledge['cos_sim_max'] = cos_sim_max

        #calculate log cosSim
        cos_sim_log = np.log2(self.df_knowledge['cos_sim_max']+1)

        self.df_knowledge['cos_sim_log'] = cos_sim_log

        #Identify outliers
        df_outliers = self.find_outliers_IQR(self.df_knowledge['cos_sim_log']).to_frame().reset_index(level=0, inplace=False)
        
        #Create df of potential answers
        df_answers = self.df_knowledge[['index','title','Content','cos_sim_max','cos_sim_log',]] \
            .sort_values(by=['cos_sim_max'], ascending = False) \
            .head(len(df_outliers['index']))
        
        return df_answers
