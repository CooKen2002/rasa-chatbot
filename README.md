"# rasa-chatbot" 

first init
    py -m venv .venv_rasa
    py utils\rasa_utils.py -> train model

    optional
        create logs folder

run
    active .venv_rasa and run 2 seperate cmd:
        rasa run actions 								
		rasa run --enable-api --cors "*" -vv    (--logging logging.yml for save logs - using for scripts\user_text_from_log.py)