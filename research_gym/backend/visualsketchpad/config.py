import os


# use this after building your own server. You can also set up the server in other machines and paste them here.
SOM_ADDRESS = os.environ.get("SOM_ADDRESS","http://<server_ip>:8080")
GROUNDING_DINO_ADDRESS = os.environ.get("GROUNDING_DINO_ADDRESS","http://<server_ip>:8081")
DEPTH_ANYTHING_ADDRESS = os.environ.get("DEPTH_ANYTHING_ADDRESS","http://<server_ip>:8083")