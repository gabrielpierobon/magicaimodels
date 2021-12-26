# Import required libraries
import pandas as pd
import nest_asyncio
import numpy as np
import warnings
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Configurations
warnings.filterwarnings('ignore')

# This function takes a card and transforms it to the shape required by the models
def transform_card(insert):

    # Create the empty dataset to populate with our card
    set_df = pd.DataFrame(columns=['name', 'lang', 'released_at', 'mana_cost', 'cmc', 'type_line',
                                   'oracle_text', 'power', 'toughness', 'colors', 'color_identity',
                                   'keywords', 'legalities', 'games', 'set', 'set_name', 'set_type',
                                   'digital', 'rarity', 'flavor_text', 'artist', 'edhrec_rank', 'prices',
                                   'loyalty', 'prints','image_uris', 'card_faces', 'oracle_text_1', 'oracle_text_2',
                                   'image_uris_1', 'image_uris_2'])

    # Insert the new card into the empty dataframe from before
    set_df = set_df.append(insert,ignore_index=True)

    # If it has text in "oracle_text_1", then it's a Double Faced Card
    set_df['DFC'] = np.where(set_df['oracle_text_1'] != "None", 1, 0)

    # Transform the data in double faced cards

    # Let's first create a dataframe that just has the card name and the column 'card_faces'
    double_cards_df = set_df[['name','card_faces']].dropna()

    # We also filter it so we get cards that actually have 2 sides
    double_cards_df = double_cards_df[double_cards_df['card_faces']!="none"]

    # If we actually have information about the 2 faces, we separate them into 2 columns
    try:
        double_cards_df[['face1','face2']] = pd.DataFrame(double_cards_df['card_faces'].tolist(), index= double_cards_df.index)
    except:
        double_cards_df[['oracle_text_1','oracle_text_2']] = "None"

    # Now let's drop the column 'card_faces'
    double_cards_df.drop("card_faces",axis=1, inplace=True)

    # We now go into each key within the dictionary of face1 and face2 and separate them into columns
    try:
        double_cards_df[double_cards_df['face1'].apply(pd.Series).columns + "_1"] = double_cards_df['face1'].apply(pd.Series)
        double_cards_df[double_cards_df['face2'].apply(pd.Series).columns + "_2"] = double_cards_df['face2'].apply(pd.Series)
    except:
        pass

    # Define a list of columns we want to keep from the 2 sided cards
    cols_to_keep = ['name','oracle_text_1','oracle_text_2','image_uris_1','image_uris_2', 'colors_1',
                    'power_1', 'toughness_1', 'loyalty_1']

    # For each column in the dataframe, if it's not a selected column, we drop it
    for i in double_cards_df.columns:
        if i not in cols_to_keep:
            double_cards_df.drop(i, axis=1, inplace=True)

    # We now need to consolidate the 2 oracle texts into 1, we join them together
    double_cards_df['oracle_text_dobles'] = double_cards_df['oracle_text_1'] + "\n" + double_cards_df['oracle_text_2']

    # Reset the indexes
    double_cards_df = double_cards_df.reset_index(drop=True)

    # Merge the 2 faces info into our main df

    # We now merge them by card name
    set_df = set_df.merge(double_cards_df, on=["name","oracle_text_1","oracle_text_2"], how="left").drop("card_faces",axis=1)

    # We use this script to replace Nulls with "None"
    set_df[['oracle_text_1','oracle_text_2']] = set_df[['oracle_text_1','oracle_text_2']].fillna("None")

    try:
        set_df[['image_uris_1','image_uris_2', 'colors_1',
                'power_1', 'toughness_1','loyalty_1']] = set_df[['image_uris_1','image_uris_2', 'colors_1', 'power_1', 'toughness_1','loyalty_1']].fillna("None")
    except:
        pass

    # Now that we have our oracle text from the 2 card sides joined together, we want to use it to replace
    # the actual "oracle_text" from the original dataframe, which is actually empty

    # If oracle_text is empty (meaning it's a double faced card), we replace it with our 'oracle_text_dobles' column
    set_df['oracle_text'] = np.where(set_df['oracle_text'].isna(),set_df['oracle_text_dobles'],set_df['oracle_text'])

    # And now that column is useless so we drop it
    set_df = set_df.drop("oracle_text_dobles",axis=1)

    # We need to do the same for all the other columns. However, for these columns, we bring the results
    # of the front card:

    # Color of the card
    try:
        set_df['colors'] = np.where(set_df['colors'].isna(),set_df['colors_1'],set_df['colors'])
        set_df = set_df.drop("colors_1",axis=1)
    except:
        pass

    # Power of the creature
    try:
        set_df['power'] = np.where(set_df['power'].isna(),set_df['power_1'],set_df['power'])
        set_df = set_df.drop("power_1",axis=1)
    except:
        pass

    # Toughness of the creature
    try:
        set_df['toughness'] = np.where(set_df['toughness'].isna(),set_df['toughness_1'],set_df['toughness'])
        set_df = set_df.drop("toughness_1",axis=1)
    except:
        pass

    # Loyalty of the planeswalker
    try:
        set_df['loyalty'] = np.where(set_df['loyalty'].isna(),set_df['loyalty_1'],set_df['loyalty'])
        set_df = set_df.drop("loyalty_1",axis=1)
    except:
        pass


    # One last thing. We can create a new column that will indicate if the card is a double faced card or not
    set_df['DFC'] = np.where(set_df['oracle_text_1'] != "None", 1, 0)

    # CMC grouping

    # Create groupings for the cmc. For 7 or above, we group them together
    set_df['cmc_grp'] = np.where(set_df['cmc'] <= 6.0, (set_df['cmc'].astype('int').astype('str'))+"_drop", "7plus_drop")

    # Separate the Keywords column into unique keyword columns

    # Create a list to use as column names for the keyword columnn
    my_list = list(set_df['keywords'].apply(pd.Series).columns)
    string = 'keyword_'
    kw_list = [string + str(x+1) for x in my_list]
    print("Keyword Columns:")
    print(kw_list)

    #Apply the separation to our dataset
    set_df[kw_list] = set_df['keywords'].apply(pd.Series).fillna('99999')

    # Separate the Legalities column into unique legality columns

    #Apply the separation to our dataset
    set_df[set_df['legalities'].apply(pd.Series).columns] = set_df['legalities'].apply(pd.Series)

    # Separate the prices column into unique price columns

    #Apply the separation to our dataset
    set_df[set_df['prices'].apply(pd.Series).columns] = set_df['prices'].apply(pd.Series)

    # Let's check the shape of our dataframe once again
    print(f"Shape of dataframe: {set_df.shape}")

    # Colors

    print(f"Max colors in a card: {len(list(set_df['colors'].apply(pd.Series).fillna('99999').columns))}")

    # Lets create a dataframe that joins the colors to create all possible color combinations
    color_df = set_df['colors'].apply(pd.Series).fillna('')
    color_df['color'] = color_df.apply(lambda x: ''.join(sorted(x)), axis=1).replace('','Colorless')
    color_df = color_df[['color']]

    # Replace the "colors" column in the dataframe with our new column
    set_df['colors'] = color_df

    print(f"Different color in data: {len(set_df['colors'].unique())}")

    # Repeat the process for the "color_identity" column
    color_id_df = set_df['color_identity'].apply(pd.Series).fillna('')
    color_id_df['color_identity'] = color_id_df.apply(lambda x: ''.join(sorted(x)), axis=1).replace('','Colorless')
    color_id_df = color_id_df[['color_identity']]
    set_df['color_identity'] = color_id_df

    ### Remove useless columns

    # List of columns we no longer need
    cols_to_drop = ['keywords','legalities','games','prices','usd_etched']

    # Drop the columns
    set_df.drop(cols_to_drop,axis=1,inplace=True)

    # Creating the keywords columns

    #Lets create a sub dataframe with just the name of the card and the keyword columns
    temp_df = set_df[['name'] + kw_list]

    # We now want to melt this dataframe so we have the name repeated as many times as keywords, but just 1 keywords column
    temp_df2 = pd.melt(temp_df, id_vars=['name'], value_vars=kw_list).drop('variable',axis=1)

    # Now we can pivot this sub dataframe and get a column for each keyword, with 0s and 1s depending on each card
    kw_df = temp_df2.pivot(columns="value", values="value").fillna(0)

    try:
        kw_df = kw_df.drop('99999',axis=1)
    except:
        pass

    try:
        kw_df = kw_df.replace(regex={r'\D': 1})
    except:
        pass

    # Let's add the name of the card to this new sub dataframe
    result = pd.concat([temp_df2[['name']], kw_df], axis=1)

    # Summing and resetting index will help to condense the data
    final_df = result.groupby(['name']).sum().reset_index()

    # We can now merge this sub dataframe with our main dataframe and get all the keywords!
    set_df_kw = set_df.merge(final_df, on=['name'], how="left").drop(kw_list, axis=1)

    ### Replace nulls in `flavor_text`

    # If a card does not have a flavor text, let's put "no flavor text" instead
    set_df_kw['flavor_text'] = set_df_kw['flavor_text'].fillna("no_flavor_text")

    ### Replace nulls in `edhrec_rank`

    # If a card does not have an edhrec_rank, let's replace it with int 999999
    set_df_kw['edhrec_rank'] = set_df_kw['edhrec_rank'].fillna(999999).astype(int)

    # Separate column ``type_line``

    # We first separate the card type of the front from the card type of the back
    try:
        set_df_kw[['face','back']] = set_df_kw['type_line'].str.split(' // ',expand=True).fillna('None')
    except:
        set_df_kw[['face','back']] = [set_df_kw['type_line'],"None"]

    # We then separate the face type using the "-" as separator
    try:
        set_df_kw[['face_type','face_subtype']] = set_df_kw['face'].str.split(' — ',expand=True).fillna('None')
    except:
        set_df_kw['face_type'] = set_df_kw['face']
        set_df_kw['face_subtype'] = "None"

    # If a card has a back, we then separate the back type using the "-" as separator
    try:
        set_df_kw[['back_type','back_subtype']] = set_df_kw['back'].str.split(' — ',expand=True).fillna('None')
    except:
        set_df_kw['back_type'] = set_df_kw['back']
        set_df_kw['back_subtype'] = "None"

    # Separate ``face_type`` in each possible token

    # Let's obtain the max quantity of words within "face_type" column
    max_word_len = []
    for i in range(len(set_df_kw['face_type'].unique())):
        append_length = len(set_df_kw['face_type'].unique()[i].split())
        max_word_len.append(append_length)

    face_type_max = max(max_word_len)

    print(f"Max words in face_type: {face_type_max}")

    # Using our result of max words in face_type, create as many face_type_N columns
    face_type_cols = []
    for i in range(face_type_max):
        face_type_col = f"face_type_{i+1}"
        face_type_cols.append(face_type_col)

    # Use these columns to split the face_type column
    set_df_kw[face_type_cols] = set_df_kw['face_type'].str.split(' ',expand=True).fillna('None')

    # Separate ``face_subtype`` in each possible token

    # Let's obtain the max quantity of words within "face_subtype" column
    max_word_len = []
    for i in range(len(set_df_kw['face_subtype'].unique())):
        append_length = len(set_df_kw['face_subtype'].unique()[i].split())
        max_word_len.append(append_length)

    face_subtype_max = max(max_word_len)

    print(f"Max words in face_subtype: {face_subtype_max}")

    # Using our result of max words in face_subtype, create as many face_subtype_N columns
    face_subtype_cols = []
    for i in range(face_subtype_max):
        face_subtype_col = f"face_subtype_{i+1}"
        face_subtype_cols.append(face_subtype_col)

    # Use these columns to split the face_subtype column
    set_df_kw[face_subtype_cols] = set_df_kw['face_subtype'].str.split(' ',expand=True).fillna('None')

    # Separate ``back_type`` in each possible token

    # Let's obtain the max quantity of words within "back_type" column
    max_word_len = []
    for i in range(len(set_df_kw['back_type'].unique())):
        append_length = len(set_df_kw['back_type'].unique()[i].split())
        max_word_len.append(append_length)

    back_type_max = max(max_word_len)

    print(f"Max words in back_type: {back_type_max}")

    # Using our result of max words in back_type, create as many face_subtype_N columns
    back_type_cols = []
    for i in range(back_type_max):
        back_type_col = f"back_type_{i+1}"
        back_type_cols.append(back_type_col)

    # Use these columns to split the back_type column
    set_df_kw[back_type_cols] = set_df_kw['back_type'].str.split(' ',expand=True).fillna('None')

    # Separate ``back_subtype`` in each possible token

    # Let's obtain the max quantity of words within "back_subtype" column
    max_word_len = []
    for i in range(len(set_df_kw['back_subtype'].unique())):
        append_length = len(set_df_kw['back_subtype'].unique()[i].split())
        max_word_len.append(append_length)

    back_subtype_max = max(max_word_len)

    print(f"Max words in back_subtype: {back_subtype_max}")

    # Using our result of max words in back_subtype, create as many back_subtype_N columns
    back_subtype_cols = []
    for i in range(back_subtype_max):
        back_subtype_col = f"back_subtype_{i+1}"
        back_subtype_cols.append(back_subtype_col)

    # Use these columns to split the back_subtype column
    set_df_kw[back_subtype_cols] = set_df_kw['back_subtype'].str.split(' ',expand=True).fillna('None')

    # Abilities Count

    # Define a function that will split the oracle text using \n as delimiter
    def count_abilities(string):
        try:
            abilities_count = len(string.split('\n'))
        except:
            abilities_count = 0
        return abilities_count

    # Apply the function and create the "abilities_count" column
    set_df_kw['abilities_count'] = set_df_kw.apply(lambda x: count_abilities(x['oracle_text']),axis=1)

    # Cleave fix

    # Cleave transformation
    # If card has cleave, remove "[" and "]" and repeat the same orcale text removing whatever is between them
    try:
        set_df_kw['oracle_text'] = np.where(set_df_kw['Cleave']==1,
                                            set_df_kw['oracle_text'].str.replace("[","").str.replace("]","")+'\n'+set_df_kw['oracle_text'].str.replace(r"[\(\[].*?[\)\]] ", ""),
                                            set_df_kw['oracle_text'])
    except:
        pass

    # Monocolored, Multicolored and others

    # If color column has just 1 character, it's monocolored (eg. "B" or "W")
    set_df_kw['monocolored'] = np.where(set_df_kw['colors'].str.len() == 1,1,0)

    # If it has more than 1 charater and it does not say Colorless, then it's multicolored
    set_df_kw['multicolored'] = np.where((set_df_kw['colors'].str.len() > 1) & (set_df_kw['colors'] != "Colorless"),1,0)

    # And these other variants
    set_df_kw['two_colors'] = np.where(set_df_kw['colors'].str.len() == 2,1,0)
    set_df_kw['three_colors'] = np.where(set_df_kw['colors'].str.len() == 3,1,0)
    set_df_kw['four_colors'] = np.where(set_df_kw['colors'].str.len() == 4,1,0)
    set_df_kw['five_colors'] = np.where(set_df_kw['colors'].str.len() == 5,1,0)
    set_df_kw['colorless'] = np.where(set_df_kw['colors'] == "Colorless",1,0)

    # Devotion

    # We count how many mana symbols we find in a card CMC
    set_df_kw['mana_symbols_cost'] = set_df_kw['mana_cost'].str.count('W|U|B|R|G').fillna(0)

    # We also count how many specific mana symbols
    set_df_kw['devotion_W'] = set_df_kw['mana_cost'].str.count('W').fillna(0)
    set_df_kw['devotion_U'] = set_df_kw['mana_cost'].str.count('U').fillna(0)
    set_df_kw['devotion_B'] = set_df_kw['mana_cost'].str.count('B').fillna(0)
    set_df_kw['devotion_R'] = set_df_kw['mana_cost'].str.count('R').fillna(0)
    set_df_kw['devotion_G'] = set_df_kw['mana_cost'].str.count('G').fillna(0)

    # Prices

    # We create some columns to detect if we have missing prices
    set_df_kw['missing_usd'] = np.where(set_df_kw['usd'].isna(), 1, 0)
    set_df_kw['missing_usd_foil'] = np.where(set_df_kw['usd_foil'].isna(), 1, 0)
    set_df_kw['missing_eur'] = np.where(set_df_kw['eur'].isna(), 1, 0)
    set_df_kw['missing_eur_foil'] = np.where(set_df_kw['eur_foil'].isna(), 1, 0)
    set_df_kw['missing_tix'] = np.where(set_df_kw['tix'].isna(), 1, 0)

    # If there are missings, we fill them with 0
    set_df_kw['usd'] = set_df_kw['usd'].fillna(0)
    set_df_kw['eur'] = set_df_kw['eur'].fillna(0)
    set_df_kw['usd_foil'] = set_df_kw['usd_foil'].fillna(0)
    set_df_kw['eur_foil'] = set_df_kw['eur_foil'].fillna(0)
    set_df_kw['tix'] = set_df_kw['tix'].fillna(0)

    # Power & Toughness

    # We just want to fill NaNs with "None" to fix any card that is not a creature
    set_df_kw['power'] = set_df_kw['power'].fillna("None")

    # Loyalty

    # We just want to fill NaNs with "None" to fix any card that is not a planeswalker
    set_df_kw['loyalty'] = set_df_kw['loyalty'].fillna('None')

    # X spells

    # Create a column that is 1 if it's a card with X in it's mana cost
    set_df_kw['X_spell'] = np.where(set_df_kw['mana_cost'].str.contains('{X}'),1,0)

    # Text `(to be removed)`

    # Remove text between brackets in oracle_text
    set_df_kw['oracle_text'] = set_df_kw['oracle_text'].str.replace(r"\(.*\)","")

    # Mana symbols in oracle text

    # We create a column tha that is 1 if there are mana symbols inside the oracle text
    set_df_kw['mana_symbols_oracle'] = np.where(set_df_kw['oracle_text'].str.contains('{W}|{U}|{B}|{R}|{G}'),1,0)

    # We count how many different mana symbols are in the oracle text
    set_df_kw['mana_symbols_oracle_nbr'] = set_df_kw['oracle_text'].str.count('{W}|{U}|{B}|{R}|{G}')

    # Includes tapping ability

    # We create a column that is 1 if the card has {T} in the oracle_text
    set_df_kw['tapping_ability'] = np.where(set_df_kw['oracle_text'].str.contains('{T}'),1,0)

    # Includes multiple choice

    # We create a column that is 1 if the card has '• ' in the oracle_text
    set_df_kw['multiple_choice'] = np.where(set_df_kw['oracle_text'].str.contains('• '),1,0)

    # Replace card name

    #EXACT MATCH
    for i in range(len(set_df_kw)):
        set_df_kw.at[i,"oracle_text"] = set_df_kw.at[i,'oracle_text'].replace(set_df_kw.at[i,'name'].split(" // ")[0], 'CARDNAME')

        #this is to also replace cardnames from back cards
        try:
            set_df_kw.at[i,"oracle_text"] = set_df_kw.at[i,'oracle_text'].replace(set_df_kw.at[i,'name'].split(" // ")[1], 'CARDNAME')
        except:
            pass

    #FIRST NAME MATCH
    for i in range(len(set_df_kw)):
        set_df_kw.at[i,"oracle_text"] = set_df_kw.at[i,'oracle_text'].replace(set_df_kw.at[i,'name'].replace(",","").split(" // ")[0].split(" ")[0], 'CARDNAME')

        #this is to also replace cardnames from back cards
        try:
            set_df_kw.at[i,"oracle_text"] = set_df_kw.at[i,'oracle_text'].replace(set_df_kw.at[i,'name'].replace(",","").split(" // ")[1].split(" ")[0], 'CARDNAME')
        except:
            pass

    # Tokenize Oracle Text

    # Define a function that takes the oracle text, removes undesired characters, stopwords and tokenizes it
    def process_oracle(oracle):
        """Process oracle function.
        Input:
            oracle: a string containing an oracle
        Output:
            oracle_clean: a list of words containing the processed oracle
        """
        import string
        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')
        oracle = re.sub(r'\$\w*', '', oracle)
        oracle = re.sub(r'^RT[\s]+', '', oracle)
        oracle = re.sub(r'#', '', oracle)
        oracle = re.sub("\d+", '', oracle)
        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        oracle_tokens = tokenizer.tokenize(oracle)

        oracle_clean = []
        for word in oracle_tokens:
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation
                # oracle_clean.append(word)
                stem_word = stemmer.stem(word)  # stemming word
                oracle_clean.append(stem_word)

        return oracle_clean

    # Apply the function and create a new column
    set_df_kw['oracle_tokens'] = set_df_kw['oracle_text'].apply(lambda x: process_oracle(x))

    # Create columns for each token

    # Separate the tokens into columns
    tokens_df = set_df_kw['oracle_tokens'].apply(pd.Series).fillna("None")

    # Create a list with all the different tokens

    tokens_set_list = []

    remove_tokens = ['iii','None','•','x','c','r','−','g','iv','}:',
                     'eight','nine','ten','—','ii','u','b','w','p']

    for i in tokens_df.columns:
        tokens_set_list = list(set(tokens_set_list+list(tokens_df[i].unique())))

    tokens_set_list = [x for x in tokens_set_list if x not in remove_tokens]

    print(f"Number of tokens: {len(tokens_set_list)}")

    # Create a new df with as many columns as tokens and 1s or 0s if the card has that token or not
    empty_df = pd.DataFrame(columns=tokens_set_list)

    k = 1

    for i in empty_df.columns:
        print(f"Progress: {round(k/len(empty_df.columns),2)*100}%")
        for j in range(len(set_df_kw)):
            if i in set_df_kw.at[j,'oracle_tokens']:
                empty_df.at[j,i] = 1
            else:
                empty_df.at[j,i] = 0
        k = k + 1

    # Change the name of the columns with the token name and the "_tkn" string added
    empty_df.columns = empty_df.columns + "_tkn"

    print(f"Dataframe shape before merge: {set_df_kw.shape}")

    # Merge with main dataframe
    set_df_kw = pd.concat([set_df_kw, empty_df], axis=1)

    print(f"Dataframe shape after merge: {set_df_kw.shape}")

    # Create columns for each card type and subtype

    # Get a list of the card types and subtypes columns
    type_cols = face_type_cols + face_subtype_cols + back_type_cols + back_subtype_cols
    print(type_cols)

    # Create a sub-dataframe only with this columns
    types_df = set_df_kw[type_cols]

    # Create a list with all the different types

    type_set_list = []

    remove_types = []

    for i in types_df.columns:
        type_set_list = list(set(type_set_list+list(types_df[i].unique())))

    type_set_list = [x for x in type_set_list if x not in remove_types]

    # Create a new df with as many columns as types/subtypes and 1s or 0s if the card has that type/subtype or not
    empty_df = pd.DataFrame(columns=type_set_list)

    k = 1

    for i in empty_df.columns:
        print(f"Progress: {round(k/len(empty_df.columns),2)*100}%")
        for j in range(len(set_df_kw)):
            if i in set_df_kw.at[j,'type_line']:
                empty_df.at[j,i] = 1
            else:
                empty_df.at[j,i] = 0
        k= k + 1

    # Change the name of the columns with the type name and the "_type" string added
    empty_df.columns = empty_df.columns + "_type"

    print(f"Dataframe shape before merge: {set_df_kw.shape}")

    # Concatenate it to our main df
    set_df_kw = pd.concat([set_df_kw, empty_df], axis=1).drop(face_type_cols+face_subtype_cols+back_type_cols+back_subtype_cols,axis=1)

    print(f"Dataframe shape after merge: {set_df_kw.shape}")

    # Flavor text

    # Create a function that splits text into tokens and counts how many tokens are
    def count_elements_in_list(string):
        count = len(string.split())
        return count

    # Apply it to the flavor text
    set_df_kw['flavor_text_len'] = set_df_kw['flavor_text'].apply(lambda x: count_elements_in_list(x))

    # Create a column that is 1s if the card HAS flavor text and 0 if it doesn't
    set_df_kw['flavor_text'] = np.where(set_df_kw['flavor_text']=="no_flavor_text",0,1)

    # If the card has NO flavor text, change the flavor_text_len to 0
    set_df_kw['flavor_text_len'] = np.where(set_df_kw['flavor_text']==0,0,set_df_kw['flavor_text_len'])

    # Remove the `\n` from oracle_text

    # Just replacing "\n" with " "
    set_df_kw["oracle_text"] = set_df_kw["oracle_text"].apply(lambda x: x.replace("\n"," "))

    # Card Super Types!!!

    try:
        set_df_kw['counterspell'] = np.where((set_df_kw['counter_tkn']==1) &

                                             ((set_df_kw['oracle_text'].str.lower().str.contains("counter target")) |
                                             (set_df_kw['oracle_text'].str.lower().str.contains("counter all")) |
                                             (set_df_kw['oracle_text'].str.lower().str.contains("counter it")))
                                             ,1,0)
    except:
        set_df_kw['counterspell'] = 0

    set_df_kw['manarock'] = np.where(
                                    ((set_df_kw['tapping_ability']==1) |
                                     (set_df_kw['oracle_text']).str.lower().str.contains("tap")) &

                                    (set_df_kw['type_line']).str.lower().str.contains("artifact") &

                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"{t}: add.*?(mana of any color|mana of that color|{(.*?)})",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"{t}, tap an untapped.*?(mana of any color|mana of that color|{(.*?)})",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"{t}: choose a color",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['manadork'] = np.where(
                                    (set_df_kw['tapping_ability']==1)&
                                    (set_df_kw['manarock']!=1) &
                                    (set_df_kw['back_type']!="Land") &
                                    (set_df_kw['type_line']).str.lower().str.contains("creature") &

                                    (
                                    (set_df_kw['oracle_text_1'].str.lower().str.contains(r"{t}: add",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"{t}:.*?add one mana",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"{t}: add",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"add (one|two|three|four|five) mana",regex=True)==True)
                                    )
                                    ,1,0)

    #Regex for a word or a word with "-"
    una_palabra = "\w+"
    una_palabra_con_rayita = "\w+-\w+"
    regex_1 = f"({una_palabra}|{una_palabra_con_rayita})"

    set_df_kw['removal'] = np.where(
                                     (
                                      (set_df_kw['oracle_text'].str.lower().str.contains(f"(destroy|exile) target ({regex_1}|({regex_1}, {regex_1})|({regex_1}, {regex_1}, {regex_1})|({regex_1}, {regex_1}, {regex_1}, {regex_1})) (creature|permanent)(?! from (a|your) graveyard| card from (a|your) graveyard)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) another target (creature|permanent)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"destroy any number of target (creature|creatures|permanent|permanents)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) target (attacking|blocking|attacking or blocking) creature",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"destroy up to (one|two|three) target (\w+) (creature|permanent|creatures|permanents)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"exile up to (one|two|three) target (creature|permanent|creatures|permanents)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"exile up to (one|two|three) target (nonland|nonartifact) (creature|permanent|creatures|permanents)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"exile up to (one|two|three) target (\w+) (\w+) (creature|permanent|creatures|permanents)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) target (\w+) or creature",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) a (creature|permanent) with the (greatest|highest|lowest) (power|toughness|converted mana cost|mana value)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) target (creature|permanent)(?! from a graveyard| card from a graveyard)", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) up to (\w+) target (attacking or blocking|attacking|blocking) (creature|creatures)", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"target (player|opponent) sacrifices a (creature|permanent)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"each (player|opponent) sacrifices (a|one|two|three|four) (creature|creatures|permanent|permanents)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted (creature|permanent) is a treasure",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted creature doesn't untap",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(annihilator)")==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"deals damage equal to its power to target creature",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(fights|fight) target creature")==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(those creatures|the chosen creatures) fight each other",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(fights|fight) up to (\w+) target (creature|creatures)", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(fights|fight) another target creature",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"choose target creature you don't control.*?each creature.*?deals damage equal.*?to that creature",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"you may have (cardname|it) fight (that creature|target creature|another target creature)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"target creature deals damage to itself equal to (its power)",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"target creature gets -[0-9]/-[2-9]", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"target creature gets \+[0-9]/-[2-9]", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"target creature an opponent controls gets \-[0-9]/\-[2-9]", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted creature (gets|has).*?loses (all|all other) abilities", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted creature gets \-[0-9]/\-[2-9]", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted creature gets \-[0-9]/\-[2-9]", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted creature gets \+[0-9]/\-[2-9]", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(enchanted|target) creature gets \-[0-9][0-9]/\-[0-9][0-9]", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains("target creature gets \-x/\-x")==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains("target creature gets \+x/\-x")==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"target creature an opponent controls gets \-x/\-x", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted creature gets \-x/\-x", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted (creature|permanent) can't attack or block",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains("enchanted creature has defender")==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains("enchanted creature can't block.*?its activated abilities can't be activated")==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted creature.*?loses all abilities",regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted (creature|permanent) can't attack.*?block.*?and its activated abilities can't be activated", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"deals ([2-9|x]) damage.*?(creature|any target|divided as you choose|to each of them)", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"deals ([2-9|x]) damage.*?to each of up to (one|two|three|four) (target|targets)", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"deals damage equal to.*?to (any target|target creature|target attacking creature|target blocking creature|target attacking or blocking creature)", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"target creature deals (.*?) damage to itself", regex=True)==True) |
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"deals damage to (any target|target creature|target attacking creature|target blocking creature|target attacking or blocking creature).*?equal to", regex=True)==True)) &

                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(cardname|it) deals [a-zA-Z0-9] damage to that player.",regex=True)==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains("(cardname|it) deals [a-zA-Z0-9] damage to target (player|opponent) or planeswalker")==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains("(cardname|it) deals [a-zA-Z0-9] damage to that creature's controller")==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains("that was dealt damage this turn")==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains("^(?!damage|creature)\w* random")==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"search.*?(creature|artifact|enchantment) card",regex=True)==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) target land",regex=True)==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains("return it to the battlefield")==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"return that (card|creature|permanent) to the battlefield",regex=True)==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"if you control a.*?^(?!liliana)\w* planeswalker",regex=True)==False) &
                                      (set_df_kw['oracle_text'].str.lower().str.contains(r"^(?!additional cost|additional cost)\w* exile (target|a|one|two|three|all).*?from (your|a|target opponent's) graveyard",regex=True)==False)
                                      ,1,0)

    set_df_kw['wrath'] = np.where(
                                   (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) (all|all other|each|each other|all attacking) (creature|creatures|(.*?) creatures|permanent|permanents|(.*?) permanents|(nonland|multicolored) permanent|(nonland|multicolored) permanents)",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"each (creature|other creature) gets -(x|[0-9])/-(x|[2-9])", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"each creature deals damage to itself equal to", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) all artifacts, creatures, and enchantments", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"sacrifices (all|that many) (creatures|(.*?) creatures|permanents|(.*?) permanents)", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"chooses.*?then sacrifices the rest", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"creatures.*?get -(x|[0-9])/-(x|[2-9])", regex=True)==True) | #Crippling Fear
                                    (set_df_kw['oracle_text'].str.lower().str.contains(f"deals ([3-9]|x|[1-9][0-9]) damage to each (creature|{regex_1} creature)", regex=True)==True)
                                   )
                                   ,1,0)

    regex_2 = "(land|lands|basic land|basic lands|plains|island|swamp|mountain|forest|plains|islands|swamps|mountains|forests|basic plains|basic island|basic swamp|basic mountain|basic forest|basic plains|basic islands|basic swamps|basic mountains|basic forests)"

    regex_3 = "(a|one|one|two|three|up to one|up to two|up to three|up to ten|up to x|x)"

    set_df_kw['ramp'] = np.where(
                                (set_df_kw['face_type']!="Land") &
                                (set_df_kw['manadork']!=1) &
                                (set_df_kw['manarock']!=1) &
                                (set_df_kw['face_type']!="Snow Land") &
                                (set_df_kw['face_type']!="Artifact Land") &
                                (set_df_kw['type_line'].str.lower().str.contains(r"(\w+) // land", regex=True)==False) &
                                (
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"{t}: add", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"^(?!{[1-9]}: )\w* add (one|two) mana", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"{[1]}, {t}: add ({(c|w|u|b|r|g)}{(c|w|u|b|r|g)}|two)", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever enchanted land is tapped for mana.*?adds", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(f"search (your|their) library for {regex_3} {regex_2}.*?put.*?onto the battlefield", regex=True)==True)
                                ) &
                                (
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"{[1-9]}, {t}: add one mana", regex=True)==False) &
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted land.*?{t}: add {(c|1|w|u|b|r|g)}", regex=True)==False) &
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"destroy target (land|nonbasic land)", regex=True)==False) &
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"spend this mana only to", regex=True)==False)
                                )
                                ,1,0)

    set_df_kw['tutor'] = np.where(
                                (set_df_kw['ramp']!=1) &
                                (set_df_kw['face_type']!="Land") &
                                (set_df_kw['face_type']!="Snow Land") &
                                (set_df_kw['face_type']!="Artifact Land") &
                                (
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"search your (library|library and graveyard) for (a|an|any|any one|one|up to one|two|up to two|three|up to three|four|up to four|a(white|blue|black|red|green|colorless)) (card|cards|permanent|permanents|equipment|aura|aura or equipment|legendary|enchantment|enchantments|artifact|artifacts|creature|(.*?) creature cards|creature cards|creatures|sorcery|sorceries|instant|instants|planeswalker)", regex=True)==True)
                                ) &
                                (
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"put (it|them|those cards|that card) into your graveyard", regex=True)==False) &
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"named", regex=True)==False)
                                )
                                ,1,0)

    set_df_kw['cardraw'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"draw (a|one|two|three|four|five|six|seven|x|(.*?) x) (card|cards)", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"draw (cards equal to|that many cards)", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"target player draws (.*?) (card|cards)", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(look at|reveal) the.*?put.*?(into|in) your hand", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(exile|look at the).*?(card|cards).*?you may (cast|play)", regex=True)==True)
                                    ) &
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever you draw a card", regex=True)==False) &
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"if you would draw a card", regex=True)==False) &
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"draw (a|one|two|three|four) (card|cards), then discard (a|one|two|three|four) (card|cards)", regex=True)==False) &
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"discard (a|one|two|three|four) (card|cards), then draw (a|one|two|three|four) (card|cards)", regex=True)==False)
                                    )
                                    ,1,0)

    set_df_kw['burn'] = np.where(
                                (
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"deals ([1-9|x]) damage.*?(any target|player|opponent|to them|to each of them)", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"deals (x|two|three|four|five) times (damage|x damage).*?(any target|player|opponent|to them|to each of up to)", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"deals damage equal to.*?to (any target|target player|target opponent|to them|each player|each opponent)", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"deals damage to (any target|target player|target opponent|to them|each player|each opponent).*?equal to", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"deals that much damage to (any target|target player|target opponent|each player|each opponent|that source's controller)", regex=True)==True)
                                )
                                ,1,0)

    set_df_kw['discard'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(that|target|each) (player|opponent) discards (a|one|two|three|that|all|all the) (card|cards)", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"unless that player.*?discards a card", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"target (player|opponent) reveals their hand.*?you choose.*?exile (that|it)", regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['enters_bf'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(enter|enters) the battlefield", regex=True)==True)
                                    )
                                    &
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(enter|enters) the battlefield tapped", regex=True)==False) &
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"land (enter|enters) the battlefield", regex=True)==False) &
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"it becomes day", regex=True)==False) &
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"enchant creature", regex=True)==False)
                                    )
                                    ,1,0)

    set_df_kw['die_trigger'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"when (cardname|equipped creature) dies", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever.*?dies", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever.*?you (control|don't control) dies", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['attack_trigger'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(when|whenever) (cardname|equipped creature|it) attacks", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(when|whenever) (cardname|equipped creature|it) and.*?(other|another) (creature|creatures) attack", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(battalion|exert|raid)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(when|whenever) (cardname|equipped creature|it) enters the battlefield or attacks", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['pseudo_ramp'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"you may put a (land|basic land).*?onto the battlefield", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(you|each player) may (play|put) an additional land", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"if it's a land card, you may put it onto the battlefield", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"sacrifice.*?add.*?({(.*?)}|to your mana pool|mana of (any|any one) color)", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['static_ramp'] = np.where(
                                        ((set_df_kw['type_line'].str.lower().str.contains("enchantment")) |
                                        (set_df_kw['type_line'].str.lower().str.contains("creature")) |
                                        (set_df_kw['type_line'].str.lower().str.contains("artifact"))) &
                                        (set_df_kw['back'].str.lower().str.contains("land")==False) &
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"at the beginning of.*?add.*?(mana|{(.*?)})", regex=True)==True)
                                        )
                                        ,1,0)

    regex_4 = "(a|one|up to one|two|up to two|three|up to three|four|up to four|five|up to five|six|up to six|x|up to x)"

    set_df_kw['creature_tokens'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(f"(create|put) {regex_4}.*?creature (token|tokens)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(f"(living weapon|amass|fabricate|afterlife|populate)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(creature tokens|creature tokens with.*?) are created instead", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['extra_turn'] = np.where(set_df_kw['oracle_text'].str.lower().str.contains(r"(take|takes) (an|one|two) extra (turn|turns)", regex=True)==True
                                       ,1,0)

    set_df_kw['plus1_counters'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"\+1/\+1 (counter|counters)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(evolve|mentor|adapt|bolster|bloodthirst|devour|monstrosity|reinforce|training)", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['graveyard_hate'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"exile.*?from (graveyards|a graveyard|his or her graveyard|target player's graveyard|each opponent's graveyard)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"remove all graveyards from the game", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"exile.*?all (cards|creatures) from all (graveyards|opponents' hands and graveyards)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"exile each opponent's graveyard", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"if a.*?(card|creature|permanent) would (be put into.*?graveyard|die).*?(instead exile|exile it instead)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"choose target card in (target opponent's|a) graveyard.*?exile (it|them)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(target|each) player puts all the cards from their graveyard", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(creature cards|permanents|creatures|permanent cards) in (graveyards|graveyards and libraries) can't enter the battlefield", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['free_spells'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(rather than pay|without paying) (its|it's|their|this spell's|the) mana cost", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"cascade", regex=True)==True)
                                        )
                                        &
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"you may pay {", regex=True)==False)
                                        )
                                        ,1,0)

    set_df_kw['bounce_spell'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"return.*?to (it's|its|their) (owner's|owners') (hand|hands)", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"owner.*?puts it.*?(top|bottom).*?library", regex=True)==True)
                                        )
                                        &
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"^(?!islands)\w* you control", regex=True)==False) &
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(when|whenever).*?dies.*?return.*?to its owner's hand", regex=True)==False) &
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"return (cardname|the exiled card) to its owner's hand", regex=True)==False) &
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever cardname.*?return it to its owner's hand", regex=True)==False)
                                        )
                                        ,1,0)

    set_df_kw['sac_outlet'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"sacrifice (a|another) (creature|permanent)", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(exploit)", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['sac_payoff'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever (you|a player) (sacrifice|sacrifices) a (creature|permanent)", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['cant_counter'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"can't be countered", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['costx_more'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(cost|costs) (.*?) more to cast", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"ward", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['costx_moreactivate'] = np.where(
                                                (
                                                (set_df_kw['oracle_text'].str.lower().str.contains(r"(cost|costs) (.*?) more to activate", regex=True)==True)
                                                )
                                                ,1,0)

    set_df_kw['costx_less'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(cost|costs) (.*?) less to cast", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['costx_lessacitivate'] = np.where(
                                                (
                                                (set_df_kw['oracle_text'].str.lower().str.contains(r"(cost|costs) (.*?) less to activate", regex=True)==True)
                                                )
                                                ,1,0)

    set_df_kw['whenever_opp'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever (an opponent|a player)", regex=True)==True)
                                        )
                                        ,1,0)

    regex_5 = "(all|each|another|another target|x|x target|a|target|any number of|one|up to one|up to one target|two|up to two|up to two target|three|up to three|up to three target|four|up to four|up to four target)"
    regex_6 = "(card|cards|creature|creatures|nonlegendary creature|creature card|creature cards|permanent|permanents|permanent card|permanent cards|land|lands|land card|land cards|instant or sorcery card|equipment card|aura card|aura or equipment card|artifact or enchantment)"

    set_df_kw['returnfrom_gy'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(fr"return {regex_5} {regex_6}.*?from your graveyard to your hand", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(fr"return cardname from your graveyard to your hand", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(fr"choose.*?graveyard.*?return.*?to your hand", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(fr"return.*?up to.*?from your graveyard to your hand", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(fr"return (target|another target).*?card from your graveyard to your hand", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['reanimation'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(fr"(return|put) {regex_5} {regex_6}.*?from (your|a) graveyard (to|onto) the battlefield", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(fr"return cardname from your graveyard to the battlefield", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(fr"choose.*?graveyard.*?return.*?to the battlefield", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(fr"return.*?up to.*?from your graveyard to the battlefield", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(fr"enchant creature card in (a|your) graveyard.*?return enchanted creature card to the battlefield under your control", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(fr"(return|returns|put) (all|any number of) (creature|permanent|enchantment|artifact|legendary permanent|legendary creature|nonlegendary creature|nonlegendary permanents|(.*?), (.*?) and (.*?)) cards.*?from (their|your|all) (graveyard|graveyards) (to|onto) the battlefield", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(fr"(return|put) (target|another target).*?card from your graveyard to the battlefield", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['castfrom_gy'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"you may cast cardname from your graveyard", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"flashback {", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"jump-start", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"escape—{", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"disturb {", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"unearth {", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"retrace", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"embalm", regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['lord'] = np.where(
                                (
                                (set_df_kw['type_line'].str.lower().str.contains("creature")) |
                                (set_df_kw['type_line'].str.lower().str.contains("artifact")) |
                                (set_df_kw['type_line'].str.lower().str.contains("enchantment"))
                                ) &
                                (
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"get \+[1-9]/\+[0-9]", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"(battle cry)", regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"each (creature|other creature).*?gets \+[1-9]/\+[0-9]", regex=True)==True)
                                )
                                &
                                (
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"until end of turn", regex=True)==False)
                                )
                                ,1,0)

    set_df_kw['upkeep_trigger'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"beginning of (your|enchanted player's|each|each player's) upkeep", regex=True)==True)
                                            )
                                            &
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"beginning of your upkeep, sacrifice cardname", regex=True)==False)
                                            )
                                            ,1,0)

    set_df_kw['endstep_trigger'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"beginning of (your|enchanted player's|each) end step", regex=True)==True)
                                            )
                                            &
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"sacrifice.*? at the beginning of your end step", regex=True)==False)
                                            )
                                            ,1,0)

    set_df_kw['landfall'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever a land enters the battlefield under your control", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"landfall", regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['combat_trigger'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"beginning of (combat|each combat)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"deals combat damage", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['life_gain'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"gain (.*?) life", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"gains (.*?) x life", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"gain life equal", regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(lifelink|extort)", regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['treasure_tokens'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(f"(create|put) {regex_4}.*?treasure (token|tokens)", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['protection'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(f"(hexproof|ward|indestructible|shroud)", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(f"can't (be|become) (the|target)", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(f"protection from", regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(f"becomes the target of a spell", regex=True)==True)
                                        )
                                        &
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"becomes the target of.*?sacrifice (it|cardname)", regex=True)==False) &
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"becomes the target of.*?shuffle.*?into its owner's library", regex=True)==False) &
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"becomes.*?with hexproof.*?until end of turn", regex=True)==False)
                                        )
                                        ,1,0)

    set_df_kw['cost_reduction'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(delve|convoke|affinity|foretell|madness|miracle|spectacle)", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"evoke", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"you may pay.*?to cast this spell", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"you may pay (.*?) rather than pay", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['mana_multipliers'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(whenever|if).*?tap (a|an) (land|permanent|nonland permanent|plains|island|swamp|mountain|forest|creature) for mana.*?add (one mana|an additional|{(.*?)})", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(whenever|if).*?tap (a|an) (land|permanent|nonland permanent|plains|island|swamp|mountain|forest|creature) for mana.*?it produces.*?instead", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['card_selection'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"scry", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"look at the top.*?bottom of your library.*?on top", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"look at the top.*?on top.*?bottom of your library", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"look at the top.*?graveyard.*?on top", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"look at the top.*?on top.*?graveyard", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"look at the top.*?you may put.*?into your graveyard", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"surveil", regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(explore|explores)", regex=True)==True)
                                            )
                                            &
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever a creature you control explores", regex=True)==False)
                                            )
                                            ,1,0)

    set_df_kw['whenever_cast'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(whenever you cast|prowess)",regex=True)==True)
                                            )
                                            &
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"you may transform", regex=True)==False)
                                            )
                                            ,1,0)

    set_df_kw['gain_control'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"gain control of",regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['unblockeable'] = np.where(
                                        (set_df_kw['type_line'].str.lower().str.contains("creature")) &
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(cardname|you control) can't be blocked",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(unblockable|shadow)",regex=True)==True)
                                        )
                                        &
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"cardname can't be blocked by", regex=True)==False)
                                        )
                                        ,1,0)

    set_df_kw['difficult_block'] = np.where(
                                            (set_df_kw['type_line'].str.lower().str.contains("creature")) &
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"cardname can't be blocked by",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(menace|first strike|flying|deathtouch|double strike|fear|intimidate)",regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['create_copy'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"create a copy of",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(that's|as) a copy of",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"copy (target|it|them|that spell|that ability)",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"you may copy",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(storm)",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"becomes a copy",regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['milling'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(mill|mills)",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"puts the top.*?of (their|his or her|your) library into (their|his or her|your) graveyard",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"exile the top (.*?) cards of (target|each) (player|opponent|players|opponents)",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(target|each) opponent exiles cards from the top of their library",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['trigger_multiplier'] = np.where(
                                                (
                                                (set_df_kw['oracle_text'].str.lower().str.contains(r"triggers (one more|an additional) time",regex=True)==True)
                                                )
                                                ,1,0)

    set_df_kw['untapper'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"untap (target|that|another|the chosen|them|all)",regex=True)==True)
                                    )
                                    &
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"gain control", regex=True)==False)
                                    )
                                    ,1,0)

    set_df_kw['static_effects'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(artifacts and creatures|creatures|permanents) (your opponents|enchanted player|you) (control|controls) (enter|lose|have|with|can't)",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"activated abilities of (artifacts|creatures).*?can't be activated",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"can't cause their controller to search their library",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"don't cause abilities to trigger",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"can't draw more than",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"only any time they could cast a sorcery",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"enchanted player",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"at the beginning of (your|each).*?(you|that player)",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(players|counters) can't",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"if (you|target opponent|a player|another player) would.*?instead",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"each (card|(.*?) card) in your (hand|graveyard).*?has",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"(each player|players|your opponents) can't cast (spells|more than)",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"is determined by their (power|toughness) rather than their (power|toughness)",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"each creature.*?assigns combat damage.*?toughness rather than its power",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"they put half that many",regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['damage_multipliers'] = np.where(
                                                (
                                                (set_df_kw['oracle_text'].str.lower().str.contains(r"it deals that much damage plus",regex=True)==True) |
                                                (set_df_kw['oracle_text'].str.lower().str.contains(r"it deals (double|triple) that damage",regex=True)==True)
                                                )
                                                ,1,0)

    set_df_kw['variable_pt'] = np.where(
                                        (set_df_kw['power'].str.lower().str.contains("\\*")) |
                                        (set_df_kw['toughness'].str.lower().str.contains("\\*"))
                                        ,1,0)

    set_df_kw['agressive'] = np.where(
                                     (set_df_kw['type_line'].str.lower().str.contains("creature")) &
                                     (
                                     (set_df_kw['oracle_text'].str.lower().str.contains(r"(haste|riot|dash)",regex=True)==True)
                                     )
                                     ,1,0)

    set_df_kw['doublers'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(put|it creates|it puts|create) twice that many",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['blinker'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"exile (up to (one|two) target|up to (one|two) other target|target|another target|any number of target) (creature|creatures|(.*?) creature|permanent|permanents|(.*?) permanent|(.*?) or creature).*?return.*?to the battlefield",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"exile (target|another target) (permanent|creature).*?return (that card|that permanent|it) to the battlefield under its owner's control",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"exile (two|three|four|five|all|each).*?you (control|own).*?then return (them|those).*?to the battlefield",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['graveyard_tutor'] = np.where(
                                            (set_df_kw['ramp']!=1) &
                                            (set_df_kw['tutor']!=1) &
                                            (set_df_kw['face_type']!="Land") &
                                            (set_df_kw['face_type']!="Snow Land") &
                                            (set_df_kw['face_type']!="Artifact Land") &
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"search your library for.*?put.*?into your graveyard", regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['play_toplibrary'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"play with the top of your library",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"you may (play|cast).*?(from the|the) top of your library",regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['life_lose'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(each opponent|each player|target opponent|target player).*?loses (.*?) life",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(afflict|extort)",regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['play_from_graveyard'] = np.where(
                                                (
                                                (set_df_kw['oracle_text'].str.lower().str.contains(r"you may (play|cast).*?(land|permanent|creature|artifact).*?from your graveyard",regex=True)==True)
                                                )
                                                ,1,0)

    set_df_kw['infect'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"infect",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['disenchant'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(destroy|exile) (target|each|every) (artifact or enchantment|artifact|enchantment)",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"destroy (x) target (artifacts or enchantments|artifacts|enchantments)",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"destroy all (artifacts or enchantments|artifacts|enchantments)",regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['venture'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"venture into the dungeon",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['animator'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"(target|another target).*?becomes a.*?creature",regex=True)==True)
                                    )
                                    &
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"copy", regex=True)==False) &
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"class", regex=True)==False))
                                    ,1,0)

    set_df_kw['wish'] = np.where(
                                (
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"you may.*?from outside the game",regex=True)==True) |
                                (set_df_kw['oracle_text'].str.lower().str.contains(r"learn",regex=True)==True)
                                )
                                ,1,0)

    set_df_kw['gy_synergies'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"gets.*?for each.*?in your graveyard",regex=True)==True) |
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"(dredge)",regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['looting_similar'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"draw (a|one|two|three|four) (card|cards), then discard (a|one|two|three|four) (card|cards)",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"discard (a|one|two|three|four) (card|cards)(,|:) (draw|then draw) (a|one|two|three|four) (card|cards)",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"create (.*?) (blood|clue) token",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"cycling",regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['cheatinto_play'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"creature.*?put (it|them) onto the battlefield",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"look at the.*?put.*?creature.*?onto the battlefield",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"you may put.*?(creature|permanent).*?onto the battlefield",regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['pumped_foreach'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"gets \+[0-9]/\+[0-9] for each",regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['ritual'] = np.where(
                                    (
                                    (set_df_kw['type_line'].str.lower().str.contains("instant")) |
                                    (set_df_kw['type_line'].str.lower().str.contains("sorcery"))
                                    ) &
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"add {(.*?)}",regex=True)==True) |
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"add (.*?) {(.*?)}",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['no_maximum'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"you have no maximum hand size",regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['wheel'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"each player.*?(discards|shuffles (his or her|their) hand and graveyard into (his or her|their) library).*?then draws seven cards",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['extra_combat'] = np.where(
                                        (
                                        (set_df_kw['oracle_text'].str.lower().str.contains(r"additional combat phase",regex=True)==True)
                                        )
                                        ,1,0)

    set_df_kw['ghostly_prison'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"creatures can't attack (you|you or planeswalkers you control) unless",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"whenever an opponent attacks (you|with creatures)",regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['land_destruction'] = np.where(
                                            (
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"destroy target (land|nonbasic land)",regex=True)==True) |
                                            (set_df_kw['oracle_text'].str.lower().str.contains(r"destroy all lands",regex=True)==True)
                                            )
                                            ,1,0)

    set_df_kw['win_game'] = np.where(
                                    (
                                    (set_df_kw['oracle_text'].str.lower().str.contains(r"you win the game",regex=True)==True)
                                    )
                                    ,1,0)

    set_df_kw['lose_game'] = np.where(
                                     (
                                     (set_df_kw['oracle_text'].str.lower().str.contains(r"you lose the game",regex=True)==True)
                                     )
                                     ,1,0)

    set_df_kw['cant_lose'] = np.where(
                                     (
                                     (set_df_kw['oracle_text'].str.lower().str.contains(r"you can't lose the game",regex=True)==True) |
                                     (set_df_kw['oracle_text'].str.lower().str.contains(r"your opponents can't win the game",regex=True)==True)
                                     )
                                     ,1,0)

    # Create Bins for Price categories

    # Convert column types of prices to float
    set_df_kw['usd'] = set_df_kw['usd'].astype(float)
    set_df_kw['eur'] = set_df_kw['eur'].astype(float)
    set_df_kw['tix'] = set_df_kw['tix'].astype(float)

    # Create 5 categories
    price_labels = ['bronze', 'silver', 'gold', 'platinum','diamond']

    # Define the cuts of each category
    usd_bins = [-1.00, 0.25, 1.00, 5.00, 10.00, 1000.00]
    eur_bins = [-1.00, 0.25, 1.00, 5.00, 10.00, 1000.00]
    tix_bins = [-1.00, 0.02, 0.05, 0.5, 1.00, 1000.00]

    # Apply them to the price columns
    set_df_kw['binusd'] = pd.cut(set_df_kw['usd'], bins=usd_bins, labels=price_labels)
    set_df_kw['bineur'] = pd.cut(set_df_kw['eur'], bins=eur_bins, labels=price_labels)
    set_df_kw['bintix'] = pd.cut(set_df_kw['tix'], bins=tix_bins, labels=price_labels)

    # Convert the categorical columns to string
    set_df_kw['binusd'] = set_df_kw['binusd'].astype(str)
    set_df_kw['bineur'] = set_df_kw['bineur'].astype(str)
    set_df_kw['bintix'] = set_df_kw['bintix'].astype(str)

    # Column that groups abilities

    # Define a list with all the super types we created
    abilities_columns = ['counterspell', 'manarock', 'manadork', 'removal', 'wrath', 'ramp', 'tutor', 'cardraw', 'burn',
                         'discard', 'enters_bf', 'die_trigger', 'attack_trigger', 'pseudo_ramp', 'static_ramp',
                         'creature_tokens', 'extra_turn', 'plus1_counters', 'graveyard_hate', 'free_spells', 'bounce_spell',
                         'sac_outlet', 'sac_payoff', 'cant_counter', 'costx_more', 'costx_moreactivate', 'costx_less',
                         'costx_lessacitivate', 'whenever_opp', 'returnfrom_gy', 'reanimation', 'castfrom_gy', 'lord',
                         'upkeep_trigger', 'endstep_trigger', 'landfall', 'combat_trigger', 'life_gain', 'treasure_tokens', 'protection',
                         'cost_reduction', 'mana_multipliers', 'card_selection', 'whenever_cast', 'gain_control',
                         'unblockeable', 'difficult_block', 'create_copy', 'milling', 'trigger_multiplier', 'untapper',
                         'static_effects', 'damage_multipliers', 'variable_pt', 'agressive', 'doublers', 'blinker',
                         'graveyard_tutor', 'play_toplibrary', 'life_lose', 'play_from_graveyard', 'infect', 'disenchant',
                         'venture', 'animator', 'wish', 'gy_synergies', 'looting_similar', 'cheatinto_play', 'pumped_foreach',
                         'ritual', 'no_maximum', 'wheel', 'extra_combat', 'ghostly_prison', 'land_destruction', 'win_game',
                         'lose_game', 'cant_lose']

    print(f"Total super abilities created: {len(abilities_columns)}")

    # Create a column that sums them for each card
    set_df_kw['total_abilites'] = set_df_kw[abilities_columns].sum(axis=1)

    # Release date columns

    # Convert the column to datetime
    set_df_kw['released_at'] = pd.to_datetime(set_df_kw['released_at'])

    # Extract year and month numbers
    set_df_kw['year'] = pd.DatetimeIndex(set_df_kw['released_at']).year.astype('str')
    set_df_kw['month'] = pd.DatetimeIndex(set_df_kw['released_at']).month.astype('str')

    # Land or No Land

    set_df_kw['land'] = np.where(
                                 (
                                  (set_df_kw['type_line'].str.lower().str.contains("land"))
                                 )
                                ,1,0)

    # Multiclass Columns

    set_df_kw['multiclass_colrs'] = set_df_kw['colors']
    set_df_kw['multiclass_rarty'] = set_df_kw['rarity']
    set_df_kw['multiclass_binusd'] = set_df_kw['binusd']
    set_df_kw['multiclass_bineur'] = set_df_kw['bineur']
    set_df_kw['multiclass_bintix'] = set_df_kw['bintix']

    # Filter out lands

    # Remove lands from our dataset
    set_df_kw = set_df_kw[set_df_kw['land']==0]

    # Delete token columns that do not have a relevant volume

    # Get a list of all the token columns
    tkn_col_list = []

    for e in list(set_df_kw.columns):
        for element in e.split():
            if element.endswith("_tkn"):
                tkn_col_list.append(element)

    print(f"Token Columns Length: {len(tkn_col_list)}")

    # Create a df to count ocurrance of each token column and filter out any that has lower than 3 ocurrances
    count_tkn_df = pd.DataFrame(set_df_kw[tkn_col_list].sum().sort_values(ascending=False))
    count_tkn_df.columns = ['count_tkns']
    count_tkn_df['tkn_column'] = count_tkn_df.index
    count_tkn_df = count_tkn_df.reset_index(drop=True)
    count_tkn_df = count_tkn_df[['tkn_column','count_tkns']]
    count_tkn_df = count_tkn_df.query("count_tkns >= 3")

    # Get a list of the ones that we will keep
    tkn_cols_to_keep = list(count_tkn_df['tkn_column'].unique())

    # Use the list to get another list of the columns we will want to REMOVE
    tkn_cols_to_drop = list(set(tkn_col_list) - set(tkn_cols_to_keep))

    print(f"Number of token columns to remove: {len(tkn_cols_to_drop)}")

    # Delete type columns that do not have a relevant volume

    # Get a list of all the type columns
    type_col_list = []

    for e in list(set_df_kw.columns):
        for element in e.split():
            if element.endswith("_type"):
                type_col_list.append(element)

    type_col_list.remove('set_type')
    type_col_list.remove('face_type')
    type_col_list.remove('back_type')

    print(f"Type Columns Length: {len(type_col_list)}")

    # Create a df to count ocurrance of each type column and filter out any that has lower than 0 ocurrances
    count_type_df = pd.DataFrame(set_df_kw[type_col_list].sum().sort_values(ascending=False))
    count_type_df.columns = ['count_type']
    count_type_df['type_column'] = count_type_df.index
    count_type_df = count_type_df.reset_index(drop=True)
    count_type_df = count_type_df[['type_column','count_type']]
    count_type_df = count_type_df.query("count_type >= 0")

    # Get a list of the ones that we will keep
    type_cols_to_keep = list(count_type_df['type_column'].unique())

    # Use the list to get another list of the columns we will want to REMOVE
    type_cols_to_drop = list(set(type_col_list) - set(type_cols_to_keep))

    print(f"Number of token columns to remove: {len(type_cols_to_drop)}")

    # FINAL SELECTION OF COLUMNS

    # These are other useless columns we will remove
    cols_to_drop = ['lang','released_at','mana_cost','type_line','set_name','set_type','digital',
                    'artist','image_uris','oracle_text_1','oracle_text_2','image_uris_1','image_uris_2','future','gladiator',
                    'penny','paupercommander','duel','oldschool','premodern','usd_foil', 'eur_foil','face','back','missing_usd_foil',
                    'missing_eur_foil','oracle_tokens','art_crop_image','border_crop_image','large_image',
                    'png_image','small_image','art_crop_image_1', 'border_crop_image_1', 'large_image_1',
                    'png_image_1', 'small_image_1', 'art_crop_image_2', 'border_crop_image_2', 'large_image_2',
                    'png_image_2', 'small_image_2', 'face_type','face_subtype','back_type', 'back_subtype',
                    'color_identity']

    # Remove the token columns we defined to drop
    for i in tkn_cols_to_drop:
        try:
            set_df_kw = set_df_kw.drop(i,axis=1)
        except:
            pass

    # Remove the type columns we defined to drop
    for i in type_cols_to_drop:
        try:
            set_df_kw = set_df_kw.drop(i,axis=1)
        except:
            pass

    # Remove the other columns we defined to drop
    for i in cols_to_drop:
        try:
            set_df_kw = set_df_kw.drop(i,axis=1)
        except:
            pass

    print(f"Final df shape: {set_df_kw.shape}")

    # One Hot Encode last categorical columns

    # Create a copy of our df
    set_df_enc = set_df_kw.copy()

    # Define the columns we need to encode
    cols_to_encode = ['power','toughness','colors','rarity','loyalty', 'alchemy','standard','historic','pioneer','modern',
                      'legacy','pauper','vintage','commander','brawl','historicbrawl','year','month', 'binusd',
                      'bineur','bintix']

    # For each column to encode...
    for i in cols_to_encode:

        # Get one hot encoding of columns B
        one_hot = pd.get_dummies(set_df_enc[i])

        # Add the column name to each new encoded column
        one_hot.columns = f'{i}_' + one_hot.columns

        # Drop original column as it is now encoded
        set_df_enc = set_df_enc.drop(i,axis = 1)

        # Join the encoded df
        set_df_enc = set_df_enc.join(one_hot)

    print(f"Encoded df shape: {set_df_enc.shape}")

    # Convert float columns to integers

    # First we convert the price columns to float
    set_df_enc['usd'] = set_df_enc['usd'].astype('float')
    set_df_enc['eur'] = set_df_enc['eur'].astype('float')
    set_df_enc['tix'] = set_df_enc['tix'].astype('float')

    # Columns that don't need to convert to int
    remove_list = ['name','oracle_text','set','usd','eur','tix','normal_image','normal_image_1','normal_image_2',
                   'multiclass_colrs','multiclass_rarty','multiclass_binusd','multiclass_bineur','multiclass_bintix',
                   'cmc_grp']

    # For each column of the dataset, we put it on a list to convert to integer unless it's a column from the remove_list

    float_to_int = list(set_df_enc.columns)

    for i in remove_list:
        try:
            float_to_int.remove(i)
        except:
            pass

    # Convert columns to integers
    for i in float_to_int:
        set_df_enc[i] = set_df_enc[i].astype('int')

    print(f"Encoded df shape: {set_df_enc.shape}")

    # Get all the required columns

    ad_data = pd.read_csv('./datasets/datasets_vow_20211220_FULL.csv')

    for i in ad_data.columns:
        if i not in set_df_enc:
            set_df_enc[i] = 0

    set_df_enc = set_df_enc[ad_data.columns]

    return set_df_enc
