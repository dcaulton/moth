import logging
import openai
import inspect

from .controller_gpt import GptController
from .controller_kbot import KbotController

logger = logging.getLogger(__name__)

# TODO derive this from directory structure
known_projects = ['phone_support', 'tmobile']

class AskController():
    def __init__(self, chat_data, request_data, session_key, project):
        self.chat_data = chat_data
        self.request_data = request_data
        self.session_key = session_key
        self.project = project
        self.kbot_controller = KbotController(project)
        self.gpt_controller = GptController(project)

    def ask(self):
        if self.project not in known_projects:
            return {
                'errors': ['unknown project specified',],
            }
        return self.ask_qelp()

    def reset_ask_session_variables(self):
        #Initialise and reset variables, run this once before starting a new chat session
        self.global_cost = [0.0]
        self.question_summary = ''
        self.history = []
        self.conversation_summary = ''
        self.transcript = ''
        self.knowledge = ''
        self.data = ''
        self.input_txt = ''
        self.list_ids = ''

    def check_for_changed_context(self):
        #Check to see if context changed before submitting the question to the CosSim KB function
        self.question_summary = self.input_txt # search criteria from new question only
        self.conversation_summary = self.chat_data.get('conversation_summary', '')
        if not self.chat_data.get('chat_history'):
            logger.info('no history for this chat yet')
        else:
            self.history = self.chat_data.get('chat_history')
        if self.conversation_summary:
            context = self.gpt_controller.same_context(self.conversation_summary, self.input_txt, self.global_cost).lower()
            if context == 'yes':
                self.question_summary = self.question_summary + ' ' + self.input_txt # search criteria from whole conversation
            else:
                self.question_summary = self.input_txt # search criteria from new question only
                self.history = []
            logger.info(f'same_context: {context}')

    def add_q_and_a_to_chat_history(self):
        #add Q&A to a list tracking the conversation
        self.history.append({"role": "user", "content": self.input_txt}) 
        self.history.append({"role": "assistant", "content": self.data}) 

    def save_conversation_data(self, transcript):
        #summarise transcription for question answer function (this is after the results to reduce wait time)
        logger.info(f'\nTHE QUESTION IS: {self.input_txt}')
        logger.info(f"I SEARCHED FOR DOCUMENTS RELATED TO: {self.chat_data['conversation_summary']}")
        logger.info(f'I REPLIED: {self.data}')
        conversation_summary = self.gpt_controller.summarise_history_3_5(transcript, self.global_cost)


    def ask_qelp(self):
        self.reset_ask_session_variables()
        self.input_txt = self.request_data.get('input_text')
        self.check_for_changed_context()
        self.chat_data['conversation_summary'] = self.gpt_controller.summarise_question(self.question_summary, self.global_cost) 
        df_answers = self.kbot_controller.K_BOT(self.chat_data['conversation_summary'], self.list_ids)
        #Convert relevant knowledge items into a 'table' to be included as context for the prompt
        self.knowledge = 'ID\tmanufacturer\toperating system\tproduct\tanswer\tsteps'
        for index, row in df_answers.iterrows():
            logger.info('TOMMY one answer row is ', row)
            self.knowledge = self.knowledge + '\n' +  row['id'] + '\t'  + row['manufacturer_label']+ '\t'  + row['os_name']+ '\t' + row['product_name']+ '\t'  + row['topic_name']+ '\t'  + row['steps_text']

        # Identify relevant knowledge IDs
        self.list_ids = self.gpt_controller.knowledge_ids(self.chat_data['conversation_summary'], self.knowledge, self.conversation_summary, self.global_cost)

        #Come up with a response to the question
        self.data = self.gpt_controller.run_prompt_3_5(self.chat_data['conversation_summary'], self.knowledge, self.conversation_summary, self.global_cost).split('\n')
        while("" in self.data):
            self.data.remove("")
        self.data = ''.join(self.data)

        #Format the list as text to feed back to GPT summary function
        x=0
        transcript =''
        for i in self.history:
            text = self.history[x]['role'] + '\t' + self.history[x]['content']
            transcript = transcript + text +'\n'
            x=x+1
        self.chat_data['chat_history'] = self.history

        self.save_conversation_data(transcript)

        logger.info(f'CONVERSATION SUMMARY: {self.conversation_summary}')
        logger.info(f'Cost: ${self.global_cost[0]}')

        return {
            'response_text': self.data,
            'ids': self.list_ids,
        }
