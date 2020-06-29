## Develop a NLP Model in Python & Deploy It with Flask, Step by Step 
https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776

Considering a system using machine learning to detect spam SMS text messages. Our ML systems workflow is like this: Train offline -> Make model available as a service -> Predict online.  
- A classifier is trained offline with spam and non-spam messages.
- The trained model is deployed as a service to serve users.
In this article, we will focus on both: building a machine learning model for spam SMS message classification, then create an API for the model, using Flask, the Python micro framework for building web applications.This API allows us to utilize the predictive capabilities through HTTP requests. 

### ML Model Building
Naive Bayes classifiers are a popular statistical technique of e-mail filtering. They typically use bag of words features to identify spam e-mail. Therefore, We’ll build a simple message classifier using Naive Bayes theorem.  

- sklearn.model_selection
- sklearn.feature_extraction.text
- sklearn.naive_bayes

After training the model, it is desirable to have a way to persist the model for future use without having to retrain.   
And the model will be served in a micro-service that expose endpoints to receive requests from client.  

### Turning the Spam Message Classifier into a Web Application  
Having prepared the code for classifying SMS messages in the previous section, we will develop a web application that consists of a simple web page with a form field that lets us enter a message. After submitting the message to the web application, it will render it on a new page which gives us a result of spam or not spam.  

The sub-directory templates is the directory in which Flask will look for static HTML files for rendering in the web browser, in our case, we have two html files: home.html and result.html.  

#### app.py
The app.py file contains the main code that will be executed by the Python interpreter to run the Flask web application, it included the ML code for classifying SMS messages.  

#### home.html
The following are the contents of the home.html file that will render a text form where a user can enter a message.  

#### style.css
In the header section of home.html, we loaded styles.cssfile. CSS is to determine how the look and feel of HTML documents. styles.css has to be saved in a sub-directory calledstatic, which is the default directory where Flask looks for static files such as CSS.  

#### result.html
we create a result.html file that will be rendered via the render_template('result.html', prediction=my_prediction) line return inside the predictfunction, which we defined in the app.py script to display the text that a user submitted via the text field.  

python app.py  
Now you could open a web browser and navigate to http://127.0.0.1:5000/  


