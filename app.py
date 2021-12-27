import streamlit as st
import json
import ast
import os
import datetime
from new_card_transformation import transform_card
import pandas as pd
import numpy as np
from functions import obtain_unique_values, load_models, predict_dummy_binary, predict_dummy_multiclass, predict_dummy_numeric
from streamlit_player import st_player
import nltk

# Download nltk english stopwords
nltk.download('stopwords')

# Show Magic's logo
st.sidebar.image('./static/Magic-The-Gathering-Logo.png', use_column_width=True)

# Title to display on the App
st.markdown("""
# Magic The Gathering
### Artificial Intelligence Models

Welcome to our **Magic: The Gathering card prediction models** with **Artificial Intelligence**!

*Our vision is to bring AI to Magic and help evolving this amazing game!*
""")

# If they check this button, we show a much detailed description
if st.checkbox('<<< HOW DOES IT WORK?'):
    st.markdown("""
    This site serves as demonstration to many Machine Learning models that are trained using the information from cards the Scryfall Magic The Gathering API.

    **The process goes something like this**: First we get all Magic cards from the Scryfall API, we then transform those cards into data prepared for machine learning,
    converting each card into hundreds of different data points. That's when we proceed to *show* that data to differe tmachine learning models alongside with the target
    we want to predict. The model will start learning patterns hidden within the data and improve with the more data we feed it, until it's able by itself to provide
    accurate predictions.

    As of today, we have 3 types of models and use 3 different algorithms:
    * **BINARY** models, **MULTICLASS** models and **NUMERIC** models, to define the type of prediction we want to do (a YES or NO question vs a specific category prediction)
    * **Logistic/Linear Regression**, **Gradient Boosting** and **Deep Learning** (Embeddings and bidirectional LSTM network)

    If you want to learn more about the process, **here's an article**

    **How do you start testing the models?**
    *  **<<<** Look at the sidebar on the left. There's where you pick a card or create a card yourself!
    *  Select the data source
        * FROM JSON: you provide the JSON file with your card (you can also download the template).
        * USE FORM: you complete a form with your own card data, starting from a template.
        * FROM DB: you load a REAL card from the app's database. You can also modify the card!
    *  **vvv** Look down below: Select the model you want to use
    *  Run the model with the **EXECUTE MODEL** button
    """)

# Embed a video tutorial
st_player("https://youtu.be/30_tT_R7qtQ") # youtube video tutorial
# st.video("./static/Media1.mp4", format="video/mp4")

link = '[YouTube Channel](https://www.youtube.com/channel/UC3__atAqSUrIMNLg_-6aJBA)'
st.markdown(link, unsafe_allow_html=True)

# Load the Card DB
card_db = pd.read_csv("./datasets/WEBAPP_datasets_vow_20211220_FULL.csv")

# Obtain the unique values from "keywords" column
keyword_unique_list = obtain_unique_values(card_db, 'keywords')

# Obtain the unique values from "set_type" column
settype_unique_list = list(card_db['set_type'].unique())

# Model selection
with open('./models/model_marketplace.json') as json_file:
    model_dict = json.load(json_file)

# Title for model section
st.write("### Use the models")

# Allow the user to select a model
model_selected = st.selectbox('Select your model', list(model_dict.keys()))

# Get all the data of the model selected
model = model_dict[model_selected]['MODEL']

# Print some description of the selected model to the user
st.write(f"**Model Family**: {model_dict[model_selected]['MODEL FAMILY']}")
st.write(f"**Model Question**: {model_dict[model_selected]['MODEL QUESTION']}")
st.write(f"**Model Description**: {model_dict[model_selected]['MODEL DESCRIPTION']}")

# Source Selection
source_selection = st.sidebar.radio('Select a new card!', ['FROM JSON', 'USE FORM', 'FROM DB'])

# FROM JSON
if source_selection == 'FROM JSON':

    # Load a new JSON with a card
    json_file = st.sidebar.file_uploader('Upload a card in JSON format')

    # Load the sample JSON to allow user to download a sample file
    with open("./dummy/sample.json", encoding="utf-8") as jsonFile:
        sample_json = json.load(jsonFile)

    # Allow the user to download the sample in JSON format
    st.sidebar.download_button('Download JSON sample', json.dumps(sample_json, ensure_ascii=False), file_name="sample.json")

    try:
        new_card = json.load(json_file)
        st.sidebar.write(new_card)
    except:
        pass

# FROM FORM
else:
    if source_selection == 'USE FORM':
        name = st.sidebar.text_input('Cardname', value="Normal Creature")
        lang = st.sidebar.selectbox('Language', sorted(card_db['lang'].unique()))
        released_at = st.sidebar.date_input('Released at', value=datetime.datetime(2021, 9, 24))
        mana_cost = st.sidebar.text_input('Mana Cost', value="{3}{W}")
        cmc = st.sidebar.number_input('CMC', value=4)
        type_line = st.sidebar.text_input('Card Type', value="Creature — Warrior")
        oracle_text = st.sidebar.text_area('Oracle Text', value="Vigilance\nWhenever cardname attacks, put a +1/+1 counter on it")
        oracle_text_1 = st.sidebar.text_area('DFC: Oracle Text Face', value="None")
        oracle_text_2 = st.sidebar.text_area('DFC: Oracle Text Back', value="None")
        power = st.sidebar.select_slider('Power', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"None"], value=5)
        toughness = st.sidebar.select_slider('Toughness', [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"None"], value=3)
        colors = st.sidebar.multiselect('Colors', ['W', 'U', 'B', 'R', 'G'], default=["W"])
        color_identity = st.sidebar.multiselect('Color Identity', ['W', 'U', 'B', 'R', 'G'], default=["W"])
        keywords = st.sidebar.multiselect('Keywords', keyword_unique_list, default=["Vigilance"])
        legality_standard = st.sidebar.select_slider('Standard Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_alchemy = st.sidebar.select_slider('Alchemy Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_future = st.sidebar.select_slider('Future Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_historic = st.sidebar.select_slider('Historic Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_gladiator = st.sidebar.select_slider('Gladiator Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_pioneer = st.sidebar.select_slider('Pioneer Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_modern = st.sidebar.select_slider('Modern Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_legacy = st.sidebar.select_slider('Legacy Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_pauper = st.sidebar.select_slider('Pauper Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="not_legal")
        legality_vintage = st.sidebar.select_slider('Vintage Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_penny = st.sidebar.select_slider('Penny Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_commander = st.sidebar.select_slider('Commander Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_brawl = st.sidebar.select_slider('Brawl Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_histbrawl = st.sidebar.select_slider('Historic Brawl Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_paupercomm = st.sidebar.select_slider('Pauper Commander Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="restricted")
        legality_duel = st.sidebar.select_slider('Duel Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="legal")
        legality_oldschool = st.sidebar.select_slider('Oldschool Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="not_legal")
        legality_premodern = st.sidebar.select_slider('Premodern Legality', ["legal","restricted","not_legal", "banned", "suspended"], value="not_legal")
        games = st.sidebar.multiselect('Games', ["arena", "paper", "mtgo"], default=["arena", "paper", "mtgo"])
        set = st.sidebar.text_input('Set', value="kld")
        set_name = st.sidebar.text_input('Set Name', value="Kaladesh")
        set_type = st.sidebar.select_slider('Set Type', settype_unique_list, value="expansion")
        digital = st.sidebar.select_slider('Digital', [True,False], value=False)
        rarity = st.sidebar.select_slider('Rarity', ['common','uncommon','rare','mythic'], value='uncommon')
        flavor_text = st.sidebar.text_area('Flavor Text', value="")
        artist = st.sidebar.text_input('Artist Name', value="Gabriel Pierobon")
        edhrec_rank = st.sidebar.number_input('EDHREC Rank', value=21000)
        price_usd = st.sidebar.number_input('USD Price',step=1.,format="%.2f", value=0.07)
        price_usdfoil = st.sidebar.number_input('USD Foil Price',step=1.,format="%.2f", value=0.13)
        price_usdetched = st.sidebar.number_input('USD Etched Foil Price',step=1.,format="%.2f", value=0.13)
        price_eur = st.sidebar.number_input('EUR Price',step=1.,format="%.2f", value=0.23)
        price_eurfoil = st.sidebar.number_input('EUR Foil Price',step=1.,format="%.2f", value=0.30)
        price_tix = st.sidebar.number_input('TIX Price',step=1.,format="%.2f", value=0.01)
        loyalty = st.sidebar.select_slider('Planeswalker Loyalty', [0,1,2,3,4,5,6,7,"None"], value="None")
        prints = st.sidebar.number_input('Prints', value=1)
        image_uris = st.sidebar.text_input('Image uris', value="None")
        image_uris_1 = st.sidebar.text_input('Image uris 1', value="None")
        image_uris_2 = st.sidebar.text_input('Image uris 2', value="None")
        card_faces = st.sidebar.text_input('Card Faces', value=None)

        new_card = {"name": name,
                  "lang": lang,
                  "released_at": str(released_at),
                  "mana_cost": mana_cost,
                  "cmc": cmc,
                  "type_line": type_line,
                  "oracle_text": oracle_text,
                  "power": str(power),
                  "toughness": str(toughness),
                  "colors": colors,
                  "color_identity": color_identity,
                  "keywords": keywords,
                  "legalities": {"standard": legality_standard,
                                 "alchemy": legality_alchemy,
                                 "future": legality_future,
                                 "historic": legality_historic,
                                 "gladiator": legality_gladiator,
                                 "pioneer": legality_pioneer,
                                 "modern": legality_modern,
                                 "legacy": legality_legacy,
                                 "pauper": legality_pauper,
                                 "vintage": legality_vintage,
                                 "penny": legality_penny,
                                 "commander": legality_commander,
                                 "brawl": legality_brawl,
                                 "historicbrawl": legality_histbrawl,
                                 "paupercommander": legality_paupercomm,
                                 "duel": legality_duel,
                                 "oldschool": legality_oldschool,
                                 "premodern": legality_premodern},
                  "games": games,
                  "set": set,
                  "set_name": set_name,
                  "set_type": set_type,
                  "digital": digital,
                  "rarity": rarity,
                  "flavor_text": flavor_text,
                  "artist": artist,
                  "edhrec_rank": edhrec_rank,
                  "prices": {"usd": price_usd,
                            "usd_foil": price_usdfoil,
                            "usd_etched": price_usdetched,
                            "eur": price_eur,
                            "eur_foil": price_eurfoil,
                            "tix": price_tix},
                  "loyalty": loyalty,
                  "prints": prints,
                  "image_uris": image_uris,
                  "card_faces": card_faces,
                  "oracle_text_1": oracle_text_1,
                  "oracle_text_2": oracle_text_2,
                  "image_uris_1": image_uris_1,
                  "image_uris_2": image_uris_2}

        # Allow the user to download the card in JSON format
        st.sidebar.download_button('Download JSON', json.dumps(new_card, ensure_ascii=False), file_name="new_card.json")

        try:
            st.sidebar.write(new_card)
        except:
            pass

    # FROM DB
    else:
        if source_selection == 'FROM DB':

            # Allow the user to select the set
            selected_set = st.sidebar.selectbox('Select a set', sorted(card_db['set_name'].unique()))

            # Allow the user to select a card

            # Get the data from the selected card
            selected_card = st.sidebar.selectbox('Select your card', card_db[card_db['set_name']==selected_set]['name'].unique())

            selected_card_df = card_db[(card_db['set_name']==selected_set) & (card_db['name']==selected_card)]

            # Show the Card Picture
            try:
                st.sidebar.image(ast.literal_eval(selected_card_df['image_uris'].values[0])['large'], width=200)
            except:
                st.sidebar.image([ast.literal_eval(selected_card_df['image_uris_1'].values[0])['large'],
                          ast.literal_eval(selected_card_df['image_uris_2'].values[0])['large']], width=200)

            name = st.sidebar.text_input('Cardname', value=selected_card_df['name'].values[0])
            lang = st.sidebar.text_input('Language', value=selected_card_df['lang'].values[0])
            released_at = st.sidebar.date_input('Released at', value=datetime.datetime.strptime(selected_card_df['released_at'].values[0], '%Y-%m-%d'))
            mana_cost = st.sidebar.text_input('Mana Cost', value=selected_card_df['mana_cost'].values[0])
            cmc = st.sidebar.number_input('CMC', value=int(selected_card_df['cmc'].values[0]))
            type_line = st.sidebar.text_input('Card Type', value=selected_card_df['type_line'].values[0])
            oracle_text = st.sidebar.text_area('Oracle Text', value=selected_card_df['oracle_text'].values[0])
            oracle_text_1 = st.sidebar.text_area('DFC: Oracle Text Face', value=selected_card_df['oracle_text_1'].values[0])
            oracle_text_2 = st.sidebar.text_area('DFC: Oracle Text Back', value=selected_card_df['oracle_text_2'].values[0])
            power = st.sidebar.select_slider('Power', ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',"None"], value=selected_card_df['power'].values[0])
            toughness = st.sidebar.select_slider('Toughness', ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',"None"], value=selected_card_df['toughness'].values[0])
            colors = st.sidebar.multiselect('Colors', ['W', 'U', 'B', 'R', 'G'], default=ast.literal_eval(selected_card_df['colors'].values[0]))
            color_identity = st.sidebar.multiselect('Color Identity', ['W', 'U', 'B', 'R', 'G'], default=ast.literal_eval(selected_card_df['color_identity'].values[0]))
            keywords = st.sidebar.multiselect('Keywords', keyword_unique_list, default=ast.literal_eval(selected_card_df['keywords'].values[0]))
            legality_standard = st.sidebar.select_slider('Standard Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['standard'])
            legality_alchemy = st.sidebar.select_slider('Alchemy Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['alchemy'])
            legality_future = st.sidebar.select_slider('Future Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['future'])
            legality_historic = st.sidebar.select_slider('Historic Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['historic'])
            legality_gladiator = st.sidebar.select_slider('Gladiator Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['gladiator'])
            legality_pioneer = st.sidebar.select_slider('Pioneer Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['pioneer'])
            legality_modern = st.sidebar.select_slider('Modern Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['modern'])
            legality_legacy = st.sidebar.select_slider('Legacy Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['legacy'])
            legality_pauper = st.sidebar.select_slider('Pauper Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['pauper'])
            legality_vintage = st.sidebar.select_slider('Vintage Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['vintage'])
            legality_penny = st.sidebar.select_slider('Penny Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['penny'])
            legality_commander = st.sidebar.select_slider('Commander Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['commander'])
            legality_brawl = st.sidebar.select_slider('Brawl Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['brawl'])
            legality_histbrawl = st.sidebar.select_slider('Historic Brawl Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['historicbrawl'])
            legality_paupercomm = st.sidebar.select_slider('Pauper Commander Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['paupercommander'])
            legality_duel = st.sidebar.select_slider('Duel Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['duel'])
            legality_oldschool = st.sidebar.select_slider('Oldschool Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['oldschool'])
            legality_premodern = st.sidebar.select_slider('Premodern Legality', ["legal","restricted","not_legal", "banned", "suspended"], value=ast.literal_eval(selected_card_df['legalities'].values[0])['premodern'])
            games = st.sidebar.multiselect('Games', ["arena", "paper", "mtgo"], default=ast.literal_eval(selected_card_df['games'].values[0]))
            set = st.sidebar.text_input('Set', value=selected_card_df['set'].values[0])
            set_name = st.sidebar.text_input('Set Name', value=selected_card_df['set_name'].values[0])
            set_type = st.sidebar.select_slider('Set Type', settype_unique_list, value=selected_card_df['set_type'].values[0])
            digital = st.sidebar.select_slider('Digital', [True,False], value=selected_card_df['digital'].values[0])
            rarity = st.sidebar.select_slider('Rarity', ['common','uncommon','rare','mythic'], value=selected_card_df['rarity'].values[0])
            flavor_text = st.sidebar.text_area('Flavor Text', value=selected_card_df['flavor_text'].values[0])
            artist = st.sidebar.text_input('Artist Name', value=selected_card_df['artist'].values[0])
            edhrec_rank = st.sidebar.number_input('EDHREC Rank', value=int(selected_card_df['edhrec_rank'].values[0]))
            price_usd = st.sidebar.number_input('USD Price',step=1.,format="%.2f", value=float(np.where(ast.literal_eval(selected_card_df['prices'].values[0])['usd']==None,0,ast.literal_eval(selected_card_df['prices'].values[0])['usd'])))
            price_usdfoil = st.sidebar.number_input('USD Foil Price',step=1.,format="%.2f", value=float(np.where(ast.literal_eval(selected_card_df['prices'].values[0])['usd_foil']==None,0,ast.literal_eval(selected_card_df['prices'].values[0])['usd_foil'])))
            price_usdetched = st.sidebar.number_input('USD Etched Foil Price',step=1.,format="%.2f", value=float(np.where(ast.literal_eval(selected_card_df['prices'].values[0])['usd_etched']==None,0,ast.literal_eval(selected_card_df['prices'].values[0])['usd_etched'])))
            price_eur = st.sidebar.number_input('EUR Price',step=1.,format="%.2f", value=float(np.where(ast.literal_eval(selected_card_df['prices'].values[0])['eur']==None,0,ast.literal_eval(selected_card_df['prices'].values[0])['eur'])))
            price_eurfoil = st.sidebar.number_input('EUR Foil Price',step=1.,format="%.2f", value=float(np.where(ast.literal_eval(selected_card_df['prices'].values[0])['eur_foil']==None,0,ast.literal_eval(selected_card_df['prices'].values[0])['eur_foil'])))
            price_tix = st.sidebar.number_input('TIX Price',step=1.,format="%.2f", value=float(np.where(ast.literal_eval(selected_card_df['prices'].values[0])['tix']==None,0,ast.literal_eval(selected_card_df['prices'].values[0])['tix'])))
            loyalty = st.sidebar.select_slider('Planeswalker Loyalty', ['0','1','2','3','4','5','6','7',"None"], value=selected_card_df['loyalty'].values[0])
            prints = st.sidebar.number_input('Prints', value=int(selected_card_df['prints'].values[0]))
            try:
                image_uris = st.sidebar.text_input('Image uris', value=ast.literal_eval(selected_card_df['image_uris'].values[0])['normal'])
            except:
                image_uris = st.sidebar.text_input('Image uris', value="None")
            try:
                image_uris_1 = st.sidebar.text_input('Image uris 1', value=ast.literal_eval(selected_card_df['image_uris_1'].values[0])['normal'])
            except:
                image_uris_1 = st.sidebar.text_input('Image uris_1', value="None")
            try:
                image_uris_2 = st.sidebar.text_input('Image uris 2', value=ast.literal_eval(selected_card_df['image_uris_1'].values[0])['normal'])
            except:
                image_uris_2 = st.sidebar.text_input('Image uris_2', value="None")
            card_faces = st.sidebar.text_input('Card Faces', value=None)

            new_card = {"name": name,
                      "lang": lang,
                      "released_at": str(released_at),
                      "mana_cost": mana_cost,
                      "cmc": cmc,
                      "type_line": type_line,
                      "oracle_text": oracle_text,
                      "power": str(power),
                      "toughness": str(toughness),
                      "colors": colors,
                      "color_identity": color_identity,
                      "keywords": keywords,
                      "legalities": {"standard": legality_standard,
                                     "alchemy": legality_alchemy,
                                     "future": legality_future,
                                     "historic": legality_historic,
                                     "gladiator": legality_gladiator,
                                     "pioneer": legality_pioneer,
                                     "modern": legality_modern,
                                     "legacy": legality_legacy,
                                     "pauper": legality_pauper,
                                     "vintage": legality_vintage,
                                     "penny": legality_penny,
                                     "commander": legality_commander,
                                     "brawl": legality_brawl,
                                     "historicbrawl": legality_histbrawl,
                                     "paupercommander": legality_paupercomm,
                                     "duel": legality_duel,
                                     "oldschool": legality_oldschool,
                                     "premodern": legality_premodern},
                      "games": games,
                      "set": set,
                      "set_name": set_name,
                      "set_type": set_type,
                      "digital": digital,
                      "rarity": rarity,
                      "flavor_text": flavor_text,
                      "artist": artist,
                      "edhrec_rank": edhrec_rank,
                      "prices": {"usd": price_usd,
                                "usd_foil": price_usdfoil,
                                "usd_etched": price_usdetched,
                                "eur": price_eur,
                                "eur_foil": price_eurfoil,
                                "tix": price_tix},
                      "loyalty": loyalty,
                      "prints": prints,
                      "image_uris": image_uris,
                      "card_faces": card_faces,
                      "oracle_text_1": oracle_text_1,
                      "oracle_text_2": oracle_text_2,
                      "image_uris_1": image_uris_1,
                      "image_uris_2": image_uris_2}

            # Allow the user to download the card in JSON format
            st.sidebar.download_button('Download JSON', json.dumps(new_card, ensure_ascii=False), file_name="new_card.json")

            try:
                st.sidebar.write(new_card)
            except:
                pass





# Get the model type and the model target
model_type, target = (model).split("_",1)

if st.button('EXECUTE MODEL!'):

    with st.spinner('Executing...'):

        # Transform the card to the data format required
        # processed_card = transform_card(card_from_json)
        processed_card = transform_card(new_card)

        # Print it to the app
        st.write("#### Card transformed to data:")
        st.dataframe(processed_card)

        # Load the models and define the target
        lr_model, gb_model, dl_model, scaler, tokenizer, required_model_cols, max_seq_lens, selected_classes, int_to_color = load_models(model_type, target)

        # PREDICTIONS

        if model_type == "BIN":
            # Obtain the predictions BINARY
            prediction_response = predict_dummy_binary(processed_card, tokenizer, max_seq_lens, scaler, required_model_cols, lr_model, gb_model, dl_model)

        else:
            if model_type == "MULT":
                # Obtain the predictions MULTICLASS
                prediction_response = predict_dummy_multiclass(processed_card, tokenizer, max_seq_lens, scaler, required_model_cols, selected_classes, int_to_color, lr_model, gb_model, dl_model)

            else:
                if model_type == "REGR":
                    # for cmc
                    minimum_val = 0
                    # Obtain the NUMERIC
                    prediction_response = predict_dummy_numeric(processed_card, tokenizer, max_seq_lens, scaler, required_model_cols, minimum_val, lr_model, gb_model, dl_model)

                else:
                    pass

        # Print the predictions to the app

        # Predictions in Markup format
        st.write("#### Predictions:")
        for i in prediction_response:
            st.write(f"##### {i}")
            prediction_df = pd.DataFrame([prediction_response[i]])
            prediction_df.index = [i]
            st.dataframe(prediction_df)

        # Predictions in JSON Format
        st.write("#### JSON Output:")
        st.write(prediction_response)


    st.success('Done!')
