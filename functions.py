import pickle
from keras.models import model_from_json
import joblib
import nltk
import string
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import json
import ast

# Obtains unique values from a column
def obtain_unique_values(df, column):
    unique_list = []
    for i in df[column].unique():
        for j in ast.literal_eval(i):
            if j not in unique_list:
                unique_list = unique_list + [j]
    return sorted(unique_list)

# Define a function that separates the oracle text of a card into tokens, remove punctuation
# If desired, it can also remove stopwords
def clean_text(text):
    stopwords = nltk.corpus.stopwords.words('english')
    try:
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split('\W+', text)
        text = [word for word in tokens if word not in stopwords] # if we want to remove stopwords
#         text = [word for word in tokens] # if we don't want to remove stopwords
    except:
        text = 'failed'
    return text

# This function loads the model selected by the user
def load_models(model_type,target): # BIN, target_binary

    # load the machine learning models from disk
    if model_type == "REGR":
        lr_model = pickle.load(open(f'./models/{model_type}_{target}/linear_regression.sav', 'rb'))
    else:
        lr_model = pickle.load(open(f'./models/{model_type}_{target}/logistic_regression.sav', 'rb'))
    print("Loaded lr model from disk")

    gb_model = pickle.load(open(f'./models/{model_type}_{target}/gradient_boosting.sav', 'rb'))
    print("Loaded gb model from disk")

    # load the Keras model json and create model
    json_file = open(f'./models/{model_type}_{target}/keras_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    dl_model = model_from_json(loaded_model_json)
    # load weights into new model
    dl_model.load_weights(f"./models/{model_type}_{target}/keras_model.h5")
    print("Loaded dl model from disk")
    # load the deep learning scaler
    scaler = joblib.load(f"./models/{model_type}_{target}/scaler.save")
    print("Loaded scaler from disk")
    # loading tokenizer
    with open(f'./models/{model_type}_{target}/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Loaded tokenizer from disk")
    # loading X_columns
    with open(f'./models/{model_type}_{target}/required_model_cols.txt', "rb") as fp:
        required_model_cols = pickle.load(fp)
        print("Loaded X_columns from disk")
    # loading max_seq_lens
    with open(f'./models/{model_type}_{target}/max_seq_lens.txt', "rb") as fp:
        max_seq_lens = pickle.load(fp)
        print("Loaded max_seq_lens from disk")

    # If model is multiclass we need the selected_classes list and the mapping dictionary
    if model_type == "MULT":
        # Load selected_classes
        with open(f'./models/MULT_{target}/selected_classes.txt', "rb") as fp:
            selected_classes = pickle.load(fp)
            print("Loaded selected_classes from disk")
        # Load mapping dictionary
        with open(f"./models/MULT_{target}/class_mapping.json") as json_file:
            int_to_color = json.load(json_file)
            print("Loaded int_to_color dict from disk")
    # If model is not multiclass we don't need them
    else:
        selected_classes = None
        int_to_color = None

    return lr_model, gb_model, dl_model, scaler, tokenizer, required_model_cols, max_seq_lens, selected_classes, int_to_color

# This functions performs predictions using the binary models
def predict_dummy_binary(df, tokenizer, max_seq_lens, scaler, required_model_cols, lr_model, gb_model, dl_model):

    # Apply the clean_text function and obtain the token sequences required for prediction
    df['clean_text'] = df['oracle_text'].apply(lambda x: clean_text(x))
    dummy_txt = df[['clean_text']]
    dummy_seq = tokenizer.texts_to_sequences(dummy_txt['clean_text'])
    dummy_seq_padded = pad_sequences(dummy_seq, max_seq_lens)

    #Predict the class
    dummy_lr_pred = (lr_model.predict(df[required_model_cols]) > 0.5).astype('int')
    dummy_gb_pred = (gb_model.predict(df[required_model_cols]) > 0.5).astype('int')
    dummy_dl_pred = (dl_model.predict([dummy_seq_padded,scaler.transform(df[required_model_cols])]) > 0.5).astype('int')

    #Predict the probability of the class
    dummy_lr_proba = lr_model.predict_proba(df[required_model_cols])[:,1]
    dummy_gb_proba = gb_model.predict_proba(df[required_model_cols])[:,1]
    dummy_dl_proba = dl_model.predict([dummy_seq_padded,scaler.transform(df[required_model_cols])])

    #Get the average prediction
    dummy_av_proba = (dummy_lr_proba + dummy_gb_proba + dummy_dl_proba)/3
    dummy_av_pred = (dummy_av_proba > 0.5).astype('int')

    #Prepare the response
    prediction_response = {
                           "model_1": {
                                        "model": "logistic_regression",
                                        "pred": dummy_lr_pred[0],
                                        "proba": round(dummy_lr_proba[0],3)
                                      },
                           "model_2": {
                                        "model": "gradient_boosting",
                                        "pred": dummy_gb_pred[0],
                                        "proba": round(dummy_gb_proba[0],3)
                                      },
                           "model_3": {
                                        "model": "deep_learning",
                                        "pred": dummy_dl_pred[0][0],
                                        "proba": round(float(round(dummy_dl_proba[0][0],3)),3)
                                      },
                           "av_pred": {
                                        "model": "average",
                                        "pred": dummy_av_pred[0][0],
                                        "proba": round(dummy_av_proba[0][0],3)
                                      },
                           }
    print("Prediction Obtained!")

    return prediction_response

# This functions performs predictions using the numeric models
def predict_dummy_numeric(df, tokenizer, max_seq_lens, scaler, required_model_cols, minimum_val, lr_model, gb_model, dl_model):

    # Apply the clean_text function and obtain the token sequences required for prediction
    df['clean_text'] = df['oracle_text'].apply(lambda x: clean_text(x))
    dummy_txt = df[['clean_text']]
    dummy_seq = tokenizer.texts_to_sequences(dummy_txt['clean_text'])
    dummy_seq_padded = pad_sequences(dummy_seq, max_seq_lens)

    #Get only numeric columns
    df = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])

    #Predict the class
    dummy_lr_pred = lr_model.predict(df[required_model_cols]).round(3)
    dummy_gb_pred = gb_model.predict(df[required_model_cols]).round(3)
    dummy_dl_pred = dl_model.predict([dummy_seq_padded,scaler.transform(df[required_model_cols])]).round(3)

    # Fix the lower predictions (anything below 0 must be fixed)
    dummy_lr_pred = np.where(dummy_lr_pred <= 0, minimum_val, dummy_lr_pred)
    dummy_gb_pred = np.where(dummy_gb_pred <= 0, minimum_val, dummy_gb_pred)
    dummy_dl_pred = np.where(dummy_dl_pred <= 0, minimum_val, dummy_dl_pred)

    #Add a column with the predicted value
    df['lr_pred'] = dummy_lr_pred.round(3)
    df['gb_pred'] = dummy_gb_pred.round(3)
    df['dl_pred'] = dummy_dl_pred.round(3)
    df['av_pred'] = ((dummy_lr_pred+dummy_gb_pred+dummy_dl_pred)/3).round(3)

    #See the result
    df = df[['lr_pred','gb_pred','dl_pred','av_pred']]

    #Change index 0 to "prediction"
    df.index = ["prediction"]

    #Convert pandas df to dictionary for response
    prediction_response = df.to_dict()

    return prediction_response

# This functions performs predictions using the multiclass models
def predict_dummy_multiclass(df, tokenizer, max_seq_lens, scaler, required_model_cols, selected_classes, int_to_color, lr_model, gb_model, dl_model):

    # Apply the clean_text function and obtain the token sequences required for prediction
    df['clean_text'] = df['oracle_text'].apply(lambda x: clean_text(x))
    dummy_txt = df[['clean_text']]
    dummy_seq = tokenizer.texts_to_sequences(dummy_txt['clean_text'])
    dummy_seq_padded = pad_sequences(dummy_seq, max_seq_lens)

    #Get only numeric columns
    df = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])

    #Predict the class
    dummy_lr_pred = lr_model.predict(df[required_model_cols])
    dummy_gb_pred = gb_model.predict(df[required_model_cols])
    dummy_dl_pred = np.argmax(dl_model.predict([dummy_seq_padded,scaler.transform(df[required_model_cols])]))

    #Get the column names of the prediction classes
    prediction_columns = []
    for i in selected_classes:
        prediction_columns = prediction_columns + [int_to_color[str(selected_classes[i])]]

    # #Predict the probability of the classes
    dummy_lr_proba = lr_model.predict_proba(df[required_model_cols]).round(3)[0]
    dummy_gb_proba = gb_model.predict_proba(df[required_model_cols]).round(3)[0]
    dummy_dl_proba = dl_model.predict([dummy_seq_padded,scaler.transform(df[required_model_cols])]).round(3)[0]
    df[prediction_columns] = [(dummy_lr_proba+dummy_gb_proba+dummy_dl_proba).round(3)]

    #Add a column with the predicted value
    df['lr_pred'] = dummy_lr_pred
    df['gb_pred'] = dummy_gb_pred
    df['dl_pred'] = dummy_dl_pred

    # Map the integer prediction to the actual category used our mapping dictionary
    df['lr_pred'] = df['lr_pred'].astype(int).astype(str).map(int_to_color)
    df['gb_pred'] = df['gb_pred'].astype(int).astype(str).map(int_to_color)
    df['dl_pred'] = df['dl_pred'].astype(str).map(int_to_color)

    #Check in each row, the 1st, 2nd and 3rd places
    for i in range(len(df)):

        #Get the column names of the prediction classes
        prediction_columns = []
        for j in selected_classes:
            prediction_columns = prediction_columns + [int_to_color[str(selected_classes[j])]]

        # We complete a column called FIRST_PRED with the class with the highest probability
        df.at[i,'FIRST_PRED'] = df.loc[i,prediction_columns].astype(float).idxmax()
        try:
            prediction_columns.remove(df.at[i,'FIRST_PRED'])
        except:
            pass

        # We complete a column called SECOND_PRED with the class with the second highest probability
        df.at[i,'SECOND_PRED'] = df.loc[i,prediction_columns].astype(float).idxmax()
        try:
            prediction_columns.remove(df.at[i,'SECOND_PRED'])
        except:
            pass

        # We complete a column called THIRD_PRED with the class with the third highest probability
        df.at[i,'THIRD_PRED'] = df.loc[i,prediction_columns].astype(float).idxmax()

    #Get the column names of the prediction classes
    prediction_columns = []
    for j in selected_classes:
        prediction_columns = prediction_columns + [int_to_color[str(selected_classes[j])]]

    #See the result
    df = df[['lr_pred','gb_pred','dl_pred'] + prediction_columns + ['FIRST_PRED', 'SECOND_PRED', 'THIRD_PRED']]

    #Change index 0 to "prediction"
    df.index = ["prediction"]

    #Convert pandas df to dictionary for response
    prediction_response = df.to_dict()
#     prediction_response = df

    return prediction_response
