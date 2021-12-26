# Magic The Gathering
### Artificial Intelligence Models

Welcome to our **Magic: The Gathering card prediction models** with **Artificial Intelligence**!

*Our vision is to bring AI to Magic and help evolving this amazing game!*

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