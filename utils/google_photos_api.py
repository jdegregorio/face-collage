import os
import pickle
import requests
import logging
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from config import CLIENT_SECRETS_FILE, SCOPES, NGROK_URL
from flask import Flask, request
import threading
import webbrowser
import time

def authenticate_google_photos():
    creds = None
    token_pickle = os.path.join('credentials', 'token.pickle')

    if os.path.exists(token_pickle):
        with open(token_pickle, 'rb') as token:
            creds = pickle.load(token)
    else:
        # Create the flow using the client secrets file and specify the redirect URI
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=f"{NGROK_URL}/oauth2callback"
        )

        # Generate the authorization URL
        auth_url, _ = flow.authorization_url(prompt='consent')

        # Open the authorization URL in the browser
        webbrowser.open(auth_url)

        # Set up the Flask app
        app = Flask(__name__)

        @app.route('/oauth2callback')
        def oauth2callback():
            flow.fetch_token(authorization_response=request.url)
            creds = flow.credentials
            # Save the credentials for future use
            with open(token_pickle, 'wb') as token:
                pickle.dump(creds, token)
            return 'Authentication successful! You can close this window.'

        # Run the Flask app in a separate thread
        def run_flask():
            app.run(port=8080)

        threading.Thread(target=run_flask).start()

        # Wait for the user to authenticate
        while not os.path.exists(token_pickle):
            time.sleep(1)  # Wait for the token file to be created

        with open(token_pickle, 'rb') as token:
            creds = pickle.load(token)

    return creds
