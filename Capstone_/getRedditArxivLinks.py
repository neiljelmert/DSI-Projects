
# USING API (trial error)

# name (agent)					Capstone_Reddit_API
# redirect uri 					http://www.example.com/unused/redirect/uri
# Reddit Client ID 				-ngcRHubeZuRCA
# Reddit Client secret 			sYtCmS8z42zIRvXPx_TL0FgdL-A

# Connect to Reddit OAuth

import requests
import requests.auth

name = 'Capstone_Reddit_API'
client_id = '-ngcRHubeZuRCA'
client_secret = 'gko_LXELoV07ZBNUXrvWZfzE3aI'
username = 'boldbrandywine'
password = 'Drumma13!'

client_auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
post_data = {"grant_type": "password", "username": username, "password": password}
headers = {"User-Agent": "ChangeMeClient/0.1 by YourUsername"}
response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
response.json()